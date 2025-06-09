from fastapi import FastAPI, HTTPException
import threading
import logging

from config import config
from models import RequestBody, HealthResponse
from services.s3_service import S3Service
from services.weavite_service import WeaviateService
from services.llm_service import LLMService
from utils.helpers import get_top_probability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration on startup
config.validate()

# Initialize FastAPI app
app = FastAPI(
    title="Medical LLM Wrapper",
    version="1.0.0",
    description="AI-powered medical skin condition analysis service"
)

# Initialize services
s3_service = S3Service()
weaviate_service = WeaviateService()
llm_service = LLMService()


def process_medical_analysis_background(request: RequestBody):
    """Synchronous background task to process medical analysis"""
    try:
        logger.info(f"Starting background processing for request: {request.user_details}")

        # Step 1: Fetch CNN analysis results from S3
        cnn_data = s3_service.fetch_json(request.cnn_response_url)
        logger.info("CNN data fetched successfully")

        # Step 2: Extract top probability condition
        top_condition, probability = get_top_probability(cnn_data)
        logger.info(f"Top condition: {top_condition}, probability: {probability}")

        # Step 3: Search Weaviate for relevant medical information
        weaviate_response = weaviate_service.search_hybrid(top_condition)
        weaviate_content = weaviate_service.extract_content(weaviate_response)
        logger.info("Weaviate search completed")

        # Step 4: Create medical consultation prompt
        medical_prompt = llm_service.create_medical_prompt(
            request.user_details,
            top_condition,
            probability,
            weaviate_content
        )

        # Step 5: Generate LLM response
        logger.info("Generating LLM response...")
        llm_response = llm_service.generate_response(medical_prompt)
        logger.info("LLM response generated successfully")

        # Step 6: Upload response to S3
        s3_response_url = s3_service.upload_response(llm_response, request.user_details, request.cnn_response_url)
        logger.info(f"Response uploaded to S3: {s3_response_url}")

        logger.info("Background processing completed successfully")

    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}", exc_info=True)


@app.post("/process-medical-analysis")
async def process_medical_analysis(request: RequestBody):
    """Fire-and-forget endpoint to trigger medical analysis processing"""
    try:
        # Start processing in a daemon thread - returns immediately
        thread = threading.Thread(
            target=process_medical_analysis_background,
            args=(request,),
            daemon=True
        )
        thread.start()

        # Return immediately without waiting
        return {
            "status": "accepted",
            "message": "Medical analysis processing has been initiated",
            "request_id": hash(str(request.user_details))
        }

    except Exception as e:
        logger.error(f"Error initiating medical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate processing: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="Medical LLM Wrapper")


@app.get("/")
async def root():
    return {"message": "Medical LLM Wrapper API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.1", port=8000)
