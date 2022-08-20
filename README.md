# MadDog

Codebase for our EACL 2021 demo paper: [A Web-based System for Acronym Identification and Disambiguation](https://aclanthology.org/2021.eacl-demos.20.pdf) (received the Best Demo Award).

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


# Citation

If you use the code released in this repo, please cite our paper:

```
@inproceedings{pouran-ben-veyseh-etal-2021-maddog,
    title = "{M}ad{D}og: A Web-based System for Acronym Identification and Disambiguation",
    author = "Pouran Ben Veyseh, Amir  and
      Dernoncourt, Franck  and
      Chang, Walter  and
      Nguyen, Thien Huu",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-demos.20",
    doi = "10.18653/v1/2021.eacl-demos.20",
    pages = "160--167",
    abstract = "Acronyms and abbreviations are the short-form of longer phrases and they are ubiquitously employed in various types of writing. Despite their usefulness to save space in writing and reader{'}s time in reading, they also provide challenges for understanding the text especially if the acronym is not defined in the text or if it is used far from its definition in long texts. To alleviate this issue, there are considerable efforts both from the research community and software developers to build systems for identifying acronyms and finding their correct meanings in the text. However, none of the existing works provide a unified solution capable of processing acronyms in various domains and to be publicly available. Thus, we provide the first web-based acronym identification and disambiguation system which can process acronyms from various domains including scientific, biomedical, and general domains. The web-based system is publicly available at http://iq.cs.uoregon.edu:5000 and a demo video is available at https://youtu.be/IkSh7LqI42M. The system source code is also available at https://github.com/amirveyseh/MadDog.",
}
```
