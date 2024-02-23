# coin-vision

This project uses tensorflow and a coin sorter to identify and sort pennies based on 
their visibile information. This project is based on object detection models and builds
an entire pipeline to support detecting pennies.

## Pipelines

## Scripts
### Image Loader

The image loader is a uses the Google API and a custome search engine to find images using a query. This script executes the query then downloads all of the results to the workspace 'images' folder.

#### Setup

This script uses the Google API and requires an API key and Custom Search Engine ID. 

Setup the Google API Key and Custom Search Engine: 
- [Create an API project](https://console.developers.google.com)
- [Enable Custom Search API](https://console.developers.google.com/apis/library/customsearch.googleapis.com)
- [Generate API key credentials](https://console.developers.google.com/apis/credentials) 
- [Setup the custome search engine](  https://cse.google.com/cse/all) and in the web form where you create/edit your custom search engine enable "Image search" option and for "Sites to search" option select "Search the entire web but emphasize included sites".
   
**Warning the usage of the custom search engine is free but highly limited.**

### Partition

### Train

### Evaluate   



### Workspace {project name}

#### Layout

- annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

- exported-models: This folder will be used to store exported versions of our trained model(s).
- images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.
 - images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.
 - images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.
- models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.
- pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.



