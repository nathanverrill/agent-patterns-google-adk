import asyncio
import json
import math
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import warnings
warnings.filterwarnings("ignore", message=".*JSON_SCHEMA_FOR_FUNC_DECL.*")

from observability import configure_logging, attach_observability
from llm import MODEL_CONFIG, build_model, warmup
logger = logging.getLogger("ReadyNowBackend")

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the model client in the background so /api/health goes green right
    # away while the slow first call happens off the critical path.
    task = asyncio.create_task(warmup())
    yield
    task.cancel()


app = FastAPI(title="ReadyNow — FEMA Emergency Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService()

def custom_before_callback(callback_context: Any, llm_request: Any) -> None:
    try:
        if not hasattr(llm_request, "contents") or not llm_request.contents:
            return
        last_turn = llm_request.contents[-1]
        if not hasattr(last_turn, "parts") or not last_turn.parts:
            return
            
        part = last_turn.parts[0]
        user_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        
        if user_text and isinstance(user_text, str):
            logger.info(f"📝 [{getattr(callback_context, 'agent_name', 'agent')}] INTERCEPTED >> {user_text.strip()}")
            
            user_text_upper = user_text.upper()
            
            non_us_indicators = ["LONDON", "PARIS", "TOKYO", "BERLIN", "FRANCE", "UK", "EUROPE"]
            if any(indicator in user_text_upper for indicator in non_us_indicators):
                refusal = "🚨 ReadyNow Boundary Policy: I am only authorized to coordinate disaster monitoring and response maneuvers within United States territories."
                if isinstance(part, dict): part["text"] = f"Output exactly this text: {refusal}"
                else: setattr(part, "text", f"Output exactly this text: {refusal}")
                return

            off_mission_keywords = ["WRITE A POEM", "REVERSE A STRING", "DROP TABLE", "PLAY A GAME", "RECIPE"]
            if any(keyword in user_text_upper for keyword in off_mission_keywords):
                refusal = "⚠️ ReadyNow Safety Directive: As a FEMA emergency response resource, I must remain fully dedicated to active disaster management, survival logistics, and routing operations. I cannot assist with non-emergency tasks."
                if isinstance(part, dict): part["text"] = f"Output exactly this text: {refusal}"
                else: setattr(part, "text", f"Output exactly this text: {refusal}")
                return
    except Exception as e:
        logger.error(f"Callback intercept error: {e}")

def custom_after_callback(callback_context: Any, llm_response: Any) -> None:
    try:
        if hasattr(llm_response, "content") and llm_response.content:
            content = llm_response.content
            if hasattr(content, "parts") and content.parts:
                part = content.parts[0]
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    logger.info(f"🤖 [{getattr(callback_context, 'agent_name', 'agent')}] DISPATCHED >> {text.strip()}")
    except Exception:
        pass

def _request_headers() -> Dict[str, str]:
    """Identify ourselves to the free OSM/NWS services, as their policies ask."""
    contact = os.getenv("READYNOW_CONTACT", "readynow-demo@example.com")
    return {"User-Agent": f"ReadyNowEmergencyAgent/1.0 ({contact})"}


def _geocode(address: str) -> Optional[Tuple[float, float]]:
    """Resolve an address to (lat, lon), or None if nothing matched.

    Google Maps when a key is configured, Nominatim (OpenStreetMap) otherwise,
    so the tool still works with no keys beyond the model's.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        try:
            logger.info("📡 GOOGLE MAPS API: Attempting premium geocoding array resolution...")
            google_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
            data = requests.get(google_url, timeout=5).json()

            if data.get("status") == "OK":
                location = data["results"][0]["geometry"]["location"]
                lat, lon = float(location["lat"]), float(location["lng"])
                logger.info(f"🎯 GOOGLE MAPS SUCCESS: Resolved coordinates [{lat:.4f}, {lon:.4f}]")
                return lat, lon
            logger.warning(f"⚠️ GOOGLE MAPS ERROR: Status returned {data.get('status')}. Dropping to fallback matrix...")
        except Exception as google_err:
            logger.warning(f"⚠️ GOOGLE MAPS EXCEPTION: {google_err}. Dropping to fallback matrix...")

    try:
        logger.info("🌐 NOMINATIM FALLBACK: Initiating open geocoding backup array...")
        nom_url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
        results = requests.get(nom_url, headers=_request_headers(), timeout=5).json()
        if results:
            lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
            logger.info(f"🎯 NOMINATIM SUCCESS: Resolved coordinates [{lat:.4f}, {lon:.4f}]")
            return lat, lon
    except Exception as nom_err:
        logger.warning(f"⚠️ NOMINATIM EXCEPTION: {nom_err}")

    return None


def geocode_and_get_weather(address: str) -> Dict[str, Any]:
    """Retrieves geospatial coordinates and fetches active NWS weather forecasts.

    Dynamically attempts Google Maps Geocoding if an API key is available,
    falling back seamlessly to Nominatim OpenStreetMap if unauthenticated.
    """
    point = _geocode(address)
    if point is None:
        return {"error": "Target location could not be verified by any geospatial arrays."}
    lat, lon = point

    try:
        headers = _request_headers()
        nws_res = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=headers, timeout=5)
        if nws_res.status_code != 200:
            return {"error": f"Meteorological data telemetry unreachable (NWS Status {nws_res.status_code})."}

        forecast_url = nws_res.json()["properties"]["forecast"]
        forecast_res = requests.get(forecast_url, headers=headers, timeout=5)
        return {"forecast": forecast_res.json()["properties"]["periods"][0]["detailedForecast"]}
    except Exception as e:
        return {"error": f"Meteorological trace exception: {str(e)}"}


# --------------------------------------------------------------------------- #
# Evacuation routing — real roads, via OSRM on OpenStreetMap data
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM = 6371.0
EVAC_DISTANCE_KM = 40.0          # far enough to clear a local hazard
HAZARD_SANITY_KM = 200.0         # farther than this and the "hazard" geocode is junk
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _osrm_base() -> str:
    """The public demo server is fine for a demo; point at your own for real load."""
    # Compose passes unset variables through as "", so don't trust getenv's default.
    return (os.getenv("OSRM_BASE_URL", "").strip() or "https://router.project-osrm.org").rstrip("/")


def _compass(bearing: float) -> str:
    return COMPASS[int((bearing % 360) / 45.0 + 0.5) % 8]


def _destination_point(lat: float, lon: float, bearing: float, km: float) -> Tuple[float, float]:
    """Great-circle point `km` away from (lat, lon) along `bearing` degrees."""
    br, lat1, lon1 = math.radians(bearing), math.radians(lat), math.radians(lon)
    d = km / EARTH_RADIUS_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _osrm_route(origin: Tuple[float, float], dest: Tuple[float, float]) -> Optional[Dict[str, Any]]:
    """Ask OSRM for a driving route and summarise the roads it uses."""
    url = (
        f"{_osrm_base()}/route/v1/driving/"
        f"{origin[1]:.5f},{origin[0]:.5f};{dest[1]:.5f},{dest[0]:.5f}"
        "?overview=false&steps=true&alternatives=false"
    )
    try:
        payload = requests.get(url, headers=_request_headers(), timeout=8).json()
    except Exception as err:
        logger.warning(f"⚠️ OSRM EXCEPTION: {err}")
        return None

    if payload.get("code") != "Ok" or not payload.get("routes"):
        logger.warning(f"⚠️ OSRM ERROR: {payload.get('code')}")
        return None

    route = payload["routes"][0]
    roads: List[str] = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            name = (step.get("ref") or step.get("name") or "").strip()
            if name and name not in roads:
                roads.append(name)

    return {
        "distance_km": round(route["distance"] / 1000.0, 1),
        "drive_time_min": round(route["duration"] / 60.0),
        "via": " → ".join(roads[:4]) if roads else "local roads",
    }


def _locate_hazard(hazard_zone: str) -> Optional[Tuple[float, float]]:
    """Best effort at placing a hazard description on the map.

    The model tends to pass things like "Tornado warning near Carthage, MO",
    which no geocoder resolves. Retry on the place name after "near"/"around"
    before giving up.
    """
    if not hazard_zone:
        return None

    candidates = [hazard_zone]
    lowered = hazard_zone.lower()
    for marker in (" near ", " around ", " at ", " in "):
        if marker in lowered:
            candidates.append(hazard_zone[lowered.rindex(marker) + len(marker):])

    for candidate in candidates:
        candidate = candidate.strip(" .,")
        if candidate:
            point = _geocode(candidate)
            if point:
                return point
    return None


def calculate_evacuation_routes(origin: str, hazard_zone: str) -> Dict[str, Any]:
    """Computes real driving evacuation routes leading away from a hazard.

    Geocodes the origin, works out which way the hazard lies, then asks OSRM
    (routing over OpenStreetMap data) for driving routes along the headings
    that lead away from it. Returns real roads, distances and drive times.
    """
    point = _geocode(origin)
    if point is None:
        return {"error": f"Could not resolve evacuation origin '{origin}'."}
    lat, lon = point

    # Where is the hazard? Often it's a phrase like "tornado" that won't geocode,
    # or resolves somewhere absurd — in that case just route outward generally.
    hazard_bearing = None
    hazard_point = _locate_hazard(hazard_zone)
    if hazard_point:
        span = _distance_km(lat, lon, *hazard_point)
        if span <= HAZARD_SANITY_KM and span > 1.0:
            hazard_bearing = _bearing(lat, lon, *hazard_point)
            logger.info(f"🧭 HAZARD VECTOR: bearing {hazard_bearing:.0f}° ({_compass(hazard_bearing)}), {span:.0f} km out")

    # Head directly away from the hazard, plus two flanking headings as backups.
    away = (hazard_bearing + 180) % 360 if hazard_bearing is not None else 0.0
    headings = [away, (away + 60) % 360, (away - 60) % 360]

    routes: List[Dict[str, Any]] = []
    for heading in headings:
        dest = _destination_point(lat, lon, heading, EVAC_DISTANCE_KM)
        logger.info(f"🛣️ OSRM: routing {_compass(heading)} from [{lat:.4f}, {lon:.4f}]")
        leg = _osrm_route((lat, lon), dest)
        if leg:
            leg["heading"] = _compass(heading)
            routes.append(leg)
        if len(routes) == 2:
            break

    if not routes:
        # OSRM unreachable — degrade to static guidance rather than failing the turn,
        # the same way geocoding degrades to Nominatim.
        logger.warning("⚠️ OSRM UNREACHABLE: falling back to static corridor guidance")
        return {
            "status": "STATIC FALLBACK — routing service unreachable",
            "origin": origin,
            "hazard_source": hazard_zone,
            "primary_evacuation_corridor": "Move perpendicular to the hazard's path on the nearest major highway.",
            "emergency_directive": "Keep radio tuned to local frequencies. Do not traverse standing water.",
        }

    result: Dict[str, Any] = {
        "status": "ROUTE COMPILED — OSRM over OpenStreetMap",
        "origin": origin,
        "origin_coordinates": [round(lat, 4), round(lon, 4)],
        "hazard_source": hazard_zone,
        "hazard_direction": _compass(hazard_bearing) if hazard_bearing is not None else "unlocated",
        "primary_evacuation_corridor": routes[0],
        "emergency_directive": "Keep radio tuned to local frequencies. Do not traverse standing water.",
    }
    if len(routes) > 1:
        result["secondary_artery"] = routes[1]
    return result


search_agent = Agent(
    name="disaster_analyst",
    model=build_model(),
    instruction="Extract location safety parameters and retrieve raw weather patterns or route metrics using tools.",
    tools=[geocode_and_get_weather, calculate_evacuation_routes]
)

critique_agent = Agent(
    name="safety_coordinator",
    model=build_model(),
    instruction="Review tactical report content. Highlight action directives, clear up complex terminology, and verify life-safety protocols stand out."
)

refine_agent = Agent(
    name="refining_editor",
    model=build_model(),
    instruction="Combine the findings and safety guidelines into a crisp, authoritative response. Keep it clear and action-oriented."
)

answer_team = SequentialAgent(
    name="fema_response_pipeline",
    description="Sequentially fetches emergency telemetry metrics, verifies communication clarity, and publishes polished updates.",
    sub_agents=[search_agent, critique_agent, refine_agent]
)

root_agent = Agent(
    name="ReadyNow_Command_Root",
    model=build_model(),
    instruction="""You are the commanding voice of Project ReadyNow, a high-performance FEMA Emergency AI Assistant.
    Your demeanor is authoritative, highly reassuring, deeply empathetic, and clear under pressure. 
    You never engage in frivolous tasks. When users present emergency scenarios, pass them to your 'fema_response_pipeline' 
    sub-agents to compile factual data, then present the resolution as a unified commanding command interface output.""",
    sub_agents=[answer_team],
    before_model_callback=custom_before_callback,
    after_model_callback=custom_after_callback
)

configure_logging()

logger.info(f"🧠 MODEL: {MODEL_CONFIG.describe()}")

attach_observability(root_agent)

runner = Runner(
    agent=root_agent, 
    session_service=session_service,
    app_name="ReadyNowEmergencyApp"
)

@app.get("/api/health")
async def health_endpoint():
    """Confirms the engine is up and reports which model it is wired to."""
    return {
        "status": "ok",
        "model": MODEL_CONFIG.model,
        "provider": MODEL_CONFIG.provider,
        "endpoint": MODEL_CONFIG.api_base or "provider default",
    }


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


APP_NAME = "ReadyNowEmergencyApp"

# Which rail stage in the console each agent lights up. The root agent isn't a
# stage — it's the dispatcher that hands off to the pipeline.
STAGE_BY_AGENT = {
    "disaster_analyst": "analyst",
    "safety_coordinator": "safety",
    "refining_editor": "editor",
}


async def _ensure_session(user_id: str, session_id: str) -> None:
    """Register the session container, tolerating one that already exists."""
    try:
        await session_service.create_session(
            user_id=user_id,
            session_id=session_id,
            app_name=APP_NAME,
        )
        logger.info(f"✨ Session created successfully: {session_id}")
    except Exception:
        # If it already exists, create_session might raise an error.
        # We catch it safely here because it means the slot is ready for text operations.
        pass


def _summarize(value: Any, limit: int = 400) -> Any:
    """Shrink a tool payload to something worth putting on the wire."""
    if isinstance(value, dict):
        return {k: _summarize(v, limit) for k, v in value.items()}
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


async def _run_pipeline(payload: ChatRequest, stream_tokens: bool = False):
    """Run one turn, yielding structured progress as the agents work.

    Yields dicts the SSE endpoint forwards verbatim and the plain JSON endpoint
    filters down to the final answer, so both routes are driven by exactly the
    same traversal.

    Every agent in the pipeline emits its own "final response" event, so
    stopping at the first one would return the analyst's raw findings and
    silently skip the safety review and the editor. Drain the stream and keep
    the last non-empty response — the editor's polished dispatch — which also
    lets the generator close cleanly instead of being cancelled mid-flight.
    """
    await _ensure_session(payload.user_id, payload.session_id)

    content = types.Content(role="user", parts=[types.Part(text=payload.message)])
    # Token streaming costs nothing extra but turns a 10-second wait into
    # something the operator can watch, so the console asks for it.
    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if stream_tokens else None

    final_response = ""
    current_agent = None

    async for event in runner.run_async(
        user_id=payload.user_id,
        session_id=payload.session_id,
        new_message=content,
        run_config=run_config,
    ):
        author = getattr(event, "author", None)
        if author and author != current_agent:
            current_agent = author
            yield {"type": "agent", "agent": author, "stage": STAGE_BY_AGENT.get(author)}

        # Partial events are the model's tokens arriving; forward them as-is.
        if getattr(event, "partial", False):
            if event.content and event.content.parts:
                chunk = event.content.parts[0].text
                if chunk:
                    yield {"type": "delta", "agent": author, "text": chunk}
            continue

        for call in (event.get_function_calls() or []):
            # transfer_to_agent is ADK plumbing, not work the operator cares about.
            if call.name != "transfer_to_agent":
                yield {"type": "tool", "agent": author, "tool": call.name, "args": dict(call.args or {})}

        for result in (event.get_function_responses() or []):
            if result.name != "transfer_to_agent":
                yield {
                    "type": "tool_result",
                    "agent": author,
                    "tool": result.name,
                    "result": _summarize(result.response),
                }

        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            if text and text.strip():
                final_response = text
                # Each agent's own finished output, so the console can show the
                # analyst's findings and the safety review, not just the ending.
                yield {
                    "type": "agent_done",
                    "agent": author,
                    "stage": STAGE_BY_AGENT.get(author),
                    "text": text,
                }

    if not final_response:
        final_response = "Communication stream link lost. Please check environment diagnostics."

    yield {"type": "final", "response": final_response}


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """Run a turn and return only the finished briefing."""
    try:
        final_response = ""
        async for update in _run_pipeline(payload):
            if update["type"] == "final":
                final_response = update["response"]
        return {"status": "success", "response": final_response}

    except Exception as e:
        logger.exception("Engine failure during process execution")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    """Same turn, streamed as Server-Sent Events.

    The console uses this to light up the response cluster from real handoffs
    and real tool calls, instead of animating a guess on a timer.
    """

    async def event_source():
        try:
            async for update in _run_pipeline(payload, stream_tokens=True):
                yield f"data: {json.dumps(update)}\n\n"
        except asyncio.CancelledError:
            # Operator navigated away mid-turn; nothing to report.
            raise
        except Exception as e:
            logger.exception("Engine failure during streamed execution")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # don't let a proxy sit on the stream
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)