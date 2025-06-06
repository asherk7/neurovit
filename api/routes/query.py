from fastapi import APIRouter, Request

router = APIRouter()

static_responses = {
    "Glioma Tumor": "Gliomas are a type of tumor in the brain or spine...",
    "Meningioma Tumor": "Meningiomas are usually benign tumors...",
    "No Tumor": "No tumor detected in this scan. This is a healthy result.",
    "Pituitary Tumor": "Pituitary tumors affect hormone regulation..."
}

@router.post("/query/")
async def query(request: Request):
    data = await request.json()
    query = data.get("query", "")

    for key in static_responses:
        if key.lower() in query.lower():
            return {"response": static_responses[key]}

    return {"response": "Our AI assistant is trained to provide information on brain tumors. Please upload an MRI scan for analysis."}
