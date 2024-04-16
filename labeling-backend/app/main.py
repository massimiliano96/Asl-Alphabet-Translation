import logging

from fastapi import BackgroundTasks, FastAPI, Request

from . import datamodel, model

app = FastAPI()
detector = model.Model()


@app.on_event("startup")
def startup():
    logging.basicConfig(format="%(levelname)s:\t %(message)s", level=logging.DEBUG)


@app.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    tasks = [datamodel.Task(t) for t in data.get("tasks", [])]
    background_tasks.add_task(detector.predict, tasks)
    return []


@app.get("/")
@app.get("/health")
def health():
    return {"status": "UP", "v2": False}


@app.post("/setup")
def setup(data: datamodel.Setup):
    detector.setup(data)
    return {"model_version": detector.model_version}
