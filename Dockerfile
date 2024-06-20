FROM python:3.9

EXPOSE 5000

# ***** Creating both backups (avro) and sources (csv) folders. *****

RUN mkdir /home/backups

RUN mkdir /home/sources

RUN mkdir /home/logs

# **** Adding the three csv files to the final destination in the container. *****s

#Left is your host (the path where the docker is) and right is the container's path.
# https://stackoverflow.com/questions/27068596/how-to-include-files-outside-of-dockers-build-context

ADD sources/word_to_index.pkl /home/sources/word_to_index.pkl

ADD sources/dic_tokens_reviews_w2v.pkl /home/sources/dic_tokens_reviews_w2v.pkl

ADD sources/my_model.keras /home/sources/my_model.keras


# ***** Instruction related to the installation and configuration of both Python libraries and Flask. *****

WORKDIR /app

#Left is your host (the path where the docker is) and right is the container's path.

COPY app/requirements.txt /app

RUN apt -qq -y update && apt -qq -y upgrade
	
RUN pip install --upgrade pip

RUN pip install 'keras<3.0.0' mediapipe-model-maker

RUN pip install -r requirements.txt

COPY app/app.py /app

CMD python app.py