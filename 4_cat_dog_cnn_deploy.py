import uvicorn
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps

app = FastAPI()
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    else:
        file_location = f"fastapidump/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        im  = Image.open(file_location)
        im = np.array(im)
        im = im.reshape(1,512,512,3)
        new_model = tf.keras.models.load_model('cat_dog_wild_tf_model')
        y_pred = new_model.predict(im)
        predicate_classes = np.argmax(y_pred, axis=1)
        classesDict = {
            0:'cat',1:'dog',2:'wild'
        }
        return {
            "animal_class":classesDict[np.argmax(np.bincount(predicate_classes))]
        }

@app.get("/")
async def root():
    return {"message": "System is Online!"}

if __name__ == "__main__":
    uvicorn.run(app)