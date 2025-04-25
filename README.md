# COIIC
Code for COIIC
![./Net](https://github.com/dart-into/COIIC/blob/main/framework.png)
## Usages

### Chinese medicinal herb object detection and key region cropping

If you want to use the Chinese medicinal herb object detection model, you can utilize the YOLO v5s code along with the model parameters provided in the best.pt file.

 
If you want to use the key region cropping method, you need to first obtain the label information provided by the object detection model, and then run yolov5_cropimg_via_labeltxt.py.

```
python yolov5_cropimg_via_labeltxt.py
```
 
Afterwards, you will obtain the cropped single Chinese medicinal herb images.

### Illumination consistency adjustment
  
If you want to use the illumination consistency adjustment method, you need to first select the corresponding reference image, and then run light_deal_mulitimgs_Va.py.

```
python light_deal_mulitimgs_Va.py
```
 
 Afterwards, you will obtain the image results adjusted according to the corresponding reference image.

### Chinese medicinal herb classification

 
If you want to use the voting strategy we designed for herbal medicine category determination, you need to place the images to be classified into different folders according to their categories, with each folder named after the corresponding category number. Then, run test_csv.py to generate the CSV file for testing.
```
python test_csv.py
```

Afterwards, use the generated test CSV file and the corresponding model parameters, and run the test4_dense_4kind_5vote.py file. Once the execution is successful, you will obtain the accuracy and recall rate results for the classification.
```
python test4_dense_4kind_5vote.py
```
Please note that the provided test code is based on the DenseNet169 network, and the model parameters model_4kind_Va2light2_4round_dense169_shai3_shai1_1.mdl are also based on this network. If you want to use a different network, you will need to retrain the model and adjust the network part in the test code accordingly.
