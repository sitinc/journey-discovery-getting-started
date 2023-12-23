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

To run the examples yourself, you will need to create a file, conf/dev.env with the following environment variables:

```conf/dev.env
OPENAI_API_KEY=<Your OpenAI API Key>
OPENAI_ORG=<Your OpenAI Org ID>
```

There are two notebook cells that will make use of the OpenAI API.  These will incur billable API usage for the 
configured OpenAI API key and org.  DO NOT execute the notebook code if you don't entirely understand the cost 
implications of the actions you will take.

The first cell of the notebook will generate sample transcripts.  The default configured value is 500.

The next cell that will make use the OpenAI API is the third cell which clusters utterances from transcripts and uses
the OpenAI chat completions API to suggest an intent name for the first 50 sampled utterances from each clustered 
intent.


I will be continuing to improve and expand the capabilities of this notebook and associated modules over time, without 
any commitments on functionality or timelines.
