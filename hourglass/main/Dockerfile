FROM tensorflow/tensorflow:1.8.0-gpu-py3

# Set the working directory to /app
WORKDIR /app

ADD ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt update && apt install -y libfontconfig1 libxrender1 libsm6 libxext6 python3-tk
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
ADD . /app

# Run app.py when the container launches
CMD bash run.sh

