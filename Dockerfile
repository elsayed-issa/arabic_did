FROM tensorflow/tensorflow

# copy requirements.txt
COPY requirements.txt .

RUN while read requirement; do pip install $requirement; done < requirements.txt