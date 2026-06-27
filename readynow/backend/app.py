import os
import sys
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

import warnings
warnings.filterwarnings("ignore", message=".*JSON_SCHEMA_FOR_FUNC_DECL.*")

from observability import configure_logging, attach_observability
logger = logging.getLogger("ReadyNowBackend")

MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "gemini/gemini-2.5-flash")

app = FastAPI(title="Project ReadyNow! - FEMA Emergency Assistant API")

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
                refusal = "🚨 ReadyNow! Boundary Policy: I am only authorized to coordinate disaster monitoring and response maneuvers within United States territories."
                if isinstance(part, dict): part["text"] = f"Output exactly this text: {refusal}"
                else: setattr(part, "text", f"Output exactly this text: {refusal}")
                return

            off_mission_keywords = ["WRITE A POEM", "REVERSE A STRING", "DROP TABLE", "PLAY A GAME", "RECIPE"]
            if any(keyword in user_text_upper for keyword in off_mission_keywords):
                refusal = "⚠️ ReadyNow! Safety Directive: As a FEMA emergency response resource, I must remain fully dedicated to active disaster management, survival logistics, and routing operations. I cannot assist with non-emergency tasks."
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

def geocode_and_get_weather(address: str) -> Dict[str, Any]:
    """Retrieves geospatial coordinates and fetches active NWS weather forecasts.
    
    Dynamically attempts Google Maps Geocoding if an API key is available,
    falling back seamlessly to Nominatim OpenStreetMap if unauthenticated.
    """
    headers = {"User-Agent": f"ReadyNowEmergencyAgent/1.0 ({os.getenv('QWIKLABS_USER', 'student-fema-session@qwiklabs.net')})"}
    api_key = os.getenv("GOOGLE_API_KEY")
    lat, lon = None, None

    # --- Step 1: Geocoding Phase ---
    if api_key:
        try:
            logger.info("📡 GOOGLE MAPS API: Attempting premium geocoding array resolution...")
            google_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
            g_res = requests.get(google_url, timeout=5)
            data = g_res.json()
            
            if data.get("status") == "OK":
                location = data["results"][0]["geometry"]["location"]
                lat, lon = float(location["lat"]), float(location["lng"])
                logger.info(f"🎯 GOOGLE MAPS SUCCESS: Resolved coordinates [{lat:.4f}, {lon:.4f}]")
            else:
                logger.warning(f"⚠️ GOOGLE MAPS ERROR: Status returned {data.get('status')}. Dropping to fallback matrix...")
        except Exception as google_err:
            logger.warning(f"⚠️ GOOGLE MAPS EXCEPTION: {google_err}. Dropping to fallback matrix...")

    # Fall back to Nominatim if Google geocoding was bypassed or failed
    if lat is None or lon is None:
        try:
            logger.info("🌐 NOMINATIM FALLBACK: Initiating open geocoding backup array...")
            nom_url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
            n_res = requests.get(nom_url, headers=headers, timeout=5)
            
            if n_res.json():
                data = n_res.json()[0]
                lat, lon = float(data["lat"]), float(data["lon"])
                logger.info(f"🎯 NOMINATIM SUCCESS: Resolved coordinates [{lat:.4f}, {lon:.4f}]")
            else:
                return {"error": "Target location could not be verified by any geospatial arrays."}
        except Exception as nom_err:
            return {"error": f"Geospatial resolution array failure: {str(nom_err)}"}

    # --- Step 2: Weather Telemetry Phase ---
    try:
        nws_res = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=headers, timeout=5)
        if nws_res.status_code != 200:
            return {"error": f"Meteorological data telemetry unreachable (NWS Status {nws_res.status_code})."}
        
        forecast_url = nws_res.json()["properties"]["forecast"]
        forecast_res = requests.get(forecast_url, headers=headers, timeout=5)
        return {"forecast": forecast_res.json()["properties"]["periods"][0]["detailedForecast"]}
    except Exception as e:
        return {"error": f"Meteorological trace exception: {str(e)}"}

def calculate_evacuation_routes(origin: str, hazard_zone: str) -> Dict[str, Any]:
    return {
        "status": "TACTICAL ROUTE COMPILED",
        "origin": origin,
        "hazard_source": hazard_zone,
        "primary_evacuation_corridor": "Take Interstate 44 Westbound away from the vector core.",
        "secondary_artery": "Route 66 Outbound to regional shelter staging structures.",
        "emergency_directive": "Keep radio tuned to local frequencies. Do not traverse standing water arrays."
    }

search_agent = Agent(
    name="disaster_analyst",
    model=LiteLlm(model=MODEL_NAME),
    instruction="Extract location safety parameters and retrieve raw weather patterns or route metrics using tools.",
    tools=[geocode_and_get_weather, calculate_evacuation_routes]
)

critique_agent = Agent(
    name="safety_coordinator",
    model=LiteLlm(model=MODEL_NAME),
    instruction="Review tactical report content. Highlight action directives, clear up complex terminology, and verify life-safety protocols stand out."
)

refine_agent = Agent(
    name="refining_editor",
    model=LiteLlm(model=MODEL_NAME),
    instruction="Combine the findings and safety guidelines into a crisp, authoritative response. Keep it clear and action-oriented."
)

answer_team = SequentialAgent(
    name="fema_response_pipeline",
    description="Sequentially fetches emergency telemetry metrics, verifies communication clarity, and publishes polished updates.",
    sub_agents=[search_agent, critique_agent, refine_agent]
)

root_agent = Agent(
    name="ReadyNow_Command_Root",
    model=LiteLlm(model=MODEL_NAME),
    instruction="""You are the commanding voice of Project ReadyNow!, a high-performance FEMA Emergency AI Assistant.
    Your demeanor is authoritative, highly reassuring, deeply empathetic, and clear under pressure. 
    You never engage in frivolous tasks. When users present emergency scenarios, pass them to your 'fema_response_pipeline' 
    sub-agents to compile factual data, then present the resolution as a unified commanding command interface output.""",
    sub_agents=[answer_team],
    before_model_callback=custom_before_callback,
    after_model_callback=custom_after_callback
)

configure_logging()

attach_observability(root_agent)

runner = Runner(
    agent=root_agent, 
    session_service=session_service,
    app_name="ReadyNowEmergencyApp"
)

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # Define a single source of truth for the application identifier string
        APP_NAME = "ReadyNowEmergencyApp"

        # Force initialize/ensure the session container is registered in memory
        try:
            await session_service.create_session(
                user_id=payload.user_id,
                session_id=payload.session_id,
                app_name=APP_NAME
            )
            logger.info(f"✨ Session created successfully: {payload.session_id}")
        except Exception:
            # If it already exists, create_session might raise an error.
            # We catch it safely here because it means the slot is ready for text operations.
            pass

        content = types.Content(role='user', parts=[types.Part(text=payload.message)])
        final_response = ""
        
        # Drive the workflow generator natively
        async for event in runner.run_async(
            user_id=payload.user_id, 
            session_id=payload.session_id, 
            new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                break
                
        if not final_response:
            final_response = "Communication stream link lost. Please check environment diagnostics."
            
        return {"status": "success", "response": final_response}
        
    except Exception as e:
        logger.exception("Engine failure during process execution")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)