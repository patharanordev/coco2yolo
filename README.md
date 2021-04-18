# **COCO Datasets Annotation for YOLO**

## **Preparation**

Download the annotation :

```bash
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

See more info, [here](https://gist.github.com/patharanordev/1342fa887c804d6374bc2e55c4ea5ac7)

## **Install Dependencies**

```bash
$ python3 -m venv env
```

## **Usage**

activate environment :

```bash
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```

Custom your request :

 - **catNms** - set class name in COCO dataset that you want.
 - **reCatIds** - reset class name index based on your class in YOLO.

```python
...
# Set annotation that you want
dataType='train2017' | 'val2017'
...

# get all images containing given categories, select one at random
reCatIds = [3,4]
catNms = ['person','cell phone']

...
```

Run script :

```bash
(env) $ python pycoco.py
```

Result is in `output` :

```
repo
|- ...
`- output
   |- COCO_{FILE_NAME}.jpg
   |- COCO_{FILE_NAME}.txt
   `- ...
```