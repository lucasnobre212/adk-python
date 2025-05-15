import asyncio
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.adk.runners import RunConfig, StreamingMode
from dotenv import load_dotenv

load_dotenv()


def get_weather_report(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Returns:
        dict: A dictionary containing the weather information with a 'status' key ('success' or 'error') and a 'report' key with the weather details if successful, or an 'error_message' if an error occurred.
    """
    if city.lower() == "london":
        return {
            "status": "success",
            "report": "The current weather in London is cloudy with a temperature of 18 degrees Celsius and a chance of rain.",
        }
    elif city.lower() == "paris":
        return {
            "status": "success",
            "report": "The weather in Paris is sunny with a temperature of 25 degrees Celsius.",
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


async def main():
    agent = LlmAgent(
        name="test_multiple_tool_call",
        model=LiteLlm(model="openai/gpt-4.1"),
        tools=[get_weather_report],
    )

    app_name = "test_multiple_tool_call"
    user_id = "test_user"
    session_id = "test_session"

    session_service = InMemorySessionService()

    runner = Runner(
        app_name=app_name,
        session_service=session_service,
        agent=agent,
    )

    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        run_config=RunConfig(
            streaming_mode=StreamingMode.SSE,
        ),
        new_message=Content(
            role="user",
            parts=[Part(text="How is the weather in London and Paris?")],
        ),
    ):
        print(event.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
