# Model

Implementation of the  [Label Studio ML Backend Template](https://dagshub.com/yonomitt/RelaxML)

To run the ML backend:

Run `docker-compose up --build`

## Label Studio ML Backend

To help you automate your annotation workflow, Label Studio allows you to connect your project to an ML backend. Once connected, Label Studio can send tasks to the ML backend to get predictions on your data. An annotator can then use these predictions to inform and speed up their labeling process.

The ML backend needs to run a REST API with some required endpoints

### API Endpoint Requirements

#### `/health`

**REQUIRED**

The health check is how Label Studio checks that a given ML backend is up and running. Label studio will call this endpoint many times including when connecting the backend and before each set of tasks it sends over.

The `/health` endpoint should return a JSON dictionary that has the following format:

```json
{
    "status": "UP",
    "v2": true
}
```

#### `/setup`

**REQUIRED**

The `/setup` endpoint is a way for Label Studio to pass information about the project to the ML backend. This includes information about the hostname, where Label Studio is running, access tokens, and schemas for the types of labels the project uses.

> **WARNING:** the `/setup` endpoint, like the health check, will be called many times. Do not do model initialization in your logic that receives this endpoint!

The `/setup` endpoint should return a JSON with the model version:

```json
{
    "model_version": "MyAwesomeModel:1.0"
}
```

This string can be whatever you want to help you identify which model was used to create predictions.

#### `/predict`

**REQUIRED**

`/predict` is the endpoint Label Studio will call to get prediction for one or more tasks. You can either return the predictions in the response to this request OR you can call a Label Studio endpoint to create predictions for a particular task.

The latter is prefered, as the former can lead to timeout errors on Label Studio's side as it waits for the response to the `/predict` endpoint.

Each task includes information about the data that needs to be annotated. If the data is a file (like an image or audio file), you will get a **repo://** URI, which can be converted to a URL for downloading.

#### `/train`

**OPTIONAL**

You can setup Label Studio to automatically train a new model after annotations have been made. This is the endpoint that will get called when that happens.

