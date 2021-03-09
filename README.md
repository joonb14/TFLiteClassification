# TensorFlow Lite Image Classification in Python

This code snipset is heavily based on <b><a href="https://www.tensorflow.org/lite/examples/image_classification/overview">TensorFlow Lite Image Classification</a></b><br>
The segmentation model can be downloaded from above link.<br>
For the realtime implementation on Android look into the <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android Image Classification Example</a><br>
Follow the <a href="https://github.com/joonb14/TFLiteClassification/blob/main/classification.ipynb">classification.ipynb</a> to get information about how to use the TFLite model in your Python environment.<br>

### Details
The <b>mobilenet_v1_1.0_224_quant.tflite</b> file's input takes normalized 224x224x3 shape image. And the output is 1001x1 where the 1001 denotes labels in below order, contains the probabilty of the image belongs to the class.. The specific labels of the 1001 classes are stored in the <b>labels_mobilenet_quant_v1_224.txt</b> file in below  order<br>
```python
lineList = ['background', 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', ...]
```
For model inference, we need to load, resize, typecast the image.<br>
The mobileNet model uses uint8 format so typecast numpy array to uint8.<br>
<img src="https://user-images.githubusercontent.com/30307587/110282617-23097d80-8022-11eb-8ca3-4bf23b1a6b68.png" width=800px/><br>
Then if you follow the correct instruction provided by Google in <a href="https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python">load_and_run_a_model_in_python</a>, you would get output in below shape<br>
<img src="https://user-images.githubusercontent.com/30307587/110282313-9eb6fa80-8021-11eb-9668-87ef47202c59.png" width=300px/><br>
Now we need to process this output to use it for classification<br>

```python
import pandas as pd
import numpy as np

classification_prob = []
classification_label = []
total = 0
for index,prob in enumerate(output_data[0]):
    if prob != 0:
        classification_prob.append(prob)
        total += prob
        classification_label.append(index)
label_names = [line.rstrip('\n') for line in open("labels_mobilenet_quant_v1_224.txt")]
found_labels = np.array(label_names)[classification_label]

df = pd.DataFrame(classification_prob/total, found_labels)
sorted_df = df.sort_values(by=0,ascending=False)
sorted_df
```
<img src="https://user-images.githubusercontent.com/30307587/110282542-010ffb00-8022-11eb-8746-7047a7386787.png" width=300px/><br>

The other models such as 
```
efficientnet-lite0-fp32.tflite, 
efficientnet-lite0-int8.tflite, 
mobilenet_v1_1.0_224.tflite
``` 
are from the <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android Image Classification Example</a> Take a look at the github and consider changing the TFLite model if you want.<br>
I believe you can modify the rest of the code as you want by yourself.<br>
Thank you!<br>
