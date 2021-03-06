# Review Classification using the Yelp Data Set
##### Will other users perceive a yelp review as useful or not?
- Main.py uses yelp review and user data to train a model that classifies each review as useful or not useful to other users
- To run the code on your machine use the docker commands below.
- The feature generation involves text processing, which is computationally expensive.
- Due to limited computing power only a very small sample from the original data set is being used.

### Docker instructions

Download Main.py from this Github repo and merged_small.pkl from Google Drive (https://drive.google.com/file/d/1v1AKrZGwKtehgQwoTUF80gDnq2b7U9Xo/view?usp=sharing) and save them in a folder. Run the following commands (still work in progress):

```sh
docker pull viavia/firstrepo:final
docker run -it -v [YOUR_PATH_TO_FOLDER]:/app -w /app  final:2 bash
```
Now you can run bash commands inside the docker container. Simply run the python file in commandline fashion.

```sh
python Main.py
```
