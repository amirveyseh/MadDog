# MadDog
A Web-based System for Acronym Identification and Disambiguation

# Deployment

To deploy the MadDog server, please make sure to have at least 125 GB of disk space and 70 GB of RAM memory. This server requires python3 to run. Follow the following steps to deploy the server:

1. Clone the repository
2. Download the pre-trained models from [https://archive.org/details/MadDog-models](https://archive.org/details/MadDog-models) and extract them in the root directory of the repository
3. Install the requirements in `requirements.txt`
4. Install the package by running `pip install -e .` in the root directory of the repository
5. Change the working directory to `prototype/app` and run `python server.py`. The server will be run on port 5000.

# Demo

Find a demo video of MadDog at [here](https://www.youtube.com/watch?v=IkSh7LqI42M)

# License

MadDog is licensed under CC BY-NC-SA 4.0.
