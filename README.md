# journey-discovery-getting-started
Python notebook and modules to explore customer journey data.

## Installation

This project requires Python 3.x. Below are the instructions to set up the environment for this project.

### Installing Dependencies

First, you need to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Installing spaCy Model

After installing the required packages, you need to download the spaCy English model "en_core_web_lg". Run the following command:

```bash
python -m spacy download en_core_web_lg
```

### Additional Requirement for Windows Users
For Windows users, to install hdbscan, you may need to install the build tools from Visual Studio. This is because hdbscan requires C++ compilation which is not natively supported in Windows Python environments.

You can download the Visual Studio Build Tools from [this link](https://visualstudio.microsoft.com/downloads/). Follow the instructions to install the necessary components.

### Usage

Explore the code and outputs in [the Jupyter notebook](https://github.com/sitinc/journey-discovery-getting-started/blob/main/notes/journey-discovery-getting-started.ipynb).

The notebook should be self-explanatory in its usage through documentation.  Pay attention to portions of the notebook 
that could incur financial cost, for example generating transcripts or naming clusters with OpenAI chat completions API.

I will be continuing to improve and expand the capabilities of this notebook and associated modules over time, without 
any commitments on functionality or timelines.
