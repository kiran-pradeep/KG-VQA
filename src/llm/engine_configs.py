from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from langchain_huggingface import ChatHuggingFace
# from langchain_ibm import ChatWatsonx
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from google.oauth2 import service_account
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

rits_api_key = os.getenv("RITS_API_KEY")
rits_url = os.getenv("RITS_BASE_URL")

vllm_host = os.getenv("VLLM_HOST")
vllm_port = os.getenv("VLLM_PORT")

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WATSONX_API_KEY")
}

project_id =os.getenv("WATSONX_PROJECT_ID")

# os.environ["LANGSMITH_API_KEY"] = "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"
# os.environ["LANGSMITH_TRACING"] = "true"

if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
    aiplatform.init(
    project=GCP_PROJECT,
    location=GCP_REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
    )
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-pro": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-pro", "temperature": 0},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-pro": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-pro-002": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro-002", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-flash":{
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-flash", "temperature": 0, "safety_settings": safety_settings}
    },
    "picker_gemini_model": {
        "constructor": VertexAI,
        "params": {"model": "projects/613565144741/locations/us-central1/endpoints/7618015791069265920", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-pro-text2sql": {
        "constructor": VertexAI,
        "params": {"model": "projects/618488765595/locations/us-central1/endpoints/1743594544210903040", "temperature": 0, "safety_settings": safety_settings}
    },
    "cot_picker": {
        "constructor": VertexAI,
        "params": {"model": "projects/243839366443/locations/us-central1/endpoints/2772315215344173056", "temperature": 0, "safety_settings": safety_settings}
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    },
    "gpt-3.5-turbo-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0}
    },
    "gpt-4-1106-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-1106-preview", "temperature": 0}
    },
    "gpt-4-0125-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-0125-preview", "temperature": 0}
    },
    "gpt-4-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-turbo", "temperature": 0}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0}
    },
    "gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0}
    },
    "claude-3-opus-20240229": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-opus-20240229", "temperature": 0}
    },
    # "finetuned_nl2sql": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/NL2SQL_DeepSeek_33B",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 400,
    #         "temperature": 0,
    #         "stop": ["```\n", ";"]
    #     }
    # },
    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9p4f6Z4W",
            "max_tokens": 400,
            "temperature": 0,
            "stop": ["```\n", ";"]
        }
    },
    "column_selection_finetuning": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9t1Gcj6Y:ckpt-step-1511",
            "max_tokens": 1000,
            "temperature": 0,
            "stop": [";"]
        }
    },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/deepseek-cot",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9oKvRYet",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    # "meta-llama/Meta-Llama-3-70B-Instruct": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 600,
    #         "temperature": 0,
    #         "model_kwargs": {
    #             "stop": [""]
    #         }
    #     }
    # },
    "meta-llama/llama-3-3-70b-instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/llama-3-3-70b-instruct",
            "openai_api_key": "/",
            "openai_api_base": f'{rits_url}/llama-3-3-70b-instruct/v1',
            "max_tokens": 1024,
            "max_retries": 10,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            "default_headers": {'RITS_API_KEY': rits_api_key},
        }
    },
    "meta-llama/llama-3-1-70b-instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/llama-3-1-70b-instruct",
            "openai_api_key": "/",
            "openai_api_base": f'{rits_url}/llama-3-1-70b-instruct/v1',
            "max_tokens": 1024,
            "max_retries": 10,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            "default_headers": {'RITS_API_KEY': rits_api_key},
        }
    },
    # "meta-llama/Llama-3.1-8B-Instruct": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "meta-llama/Llama-3.1-8B-Instruct",
    #         "openai_api_key": "/",
    #         "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
    #         "max_tokens": 1024,
    #         "max_retries": 10,
    #         # "temperature": 0,
    #         # "model_kwargs": {
    #         #     "stop": [""]
    #         # },
    #         "default_headers": {'RITS_API_KEY': rits_api_key},
    #     }
    # },
    # "meta-llama/llama-3-1-70b-instruct-wxai": {
    #     "constructor": ChatWatsonx,
    #     "params": {
    #         "model_id": "meta-llama/llama-3-1-70b-instruct",
    #         "url": credentials.get("url"),
    #         "apikey": credentials.get("apikey"),
    #         "project_id":  project_id,
    #         "params": {
    #             # GenTextParamsMetaNames.REPETITION_PENALTY: 1.05,
    #             GenTextParamsMetaNames.MAX_NEW_TOKENS: 1024,
    #             GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
    #             GenTextParamsMetaNames.RANDOM_SEED: 42,
    #             # GenTextParamsMetaNames.TEMPERATURE: 0,
    #         }
    #     }
    # },
    # "meta-llama/llama-3-1-8b-instruct": {
    #     "constructor": ChatWatsonx,
    #     "params": {
    #         "model_id": "meta-llama/llama-3-1-8b-instruct",
    #         "url": credentials.get("url"),
    #         "apikey": credentials.get("apikey"),
    #         "project_id":  project_id,
    #         "params": {
    #             # GenTextParamsMetaNames.REPETITION_PENALTY: 1.05,
    #             GenTextParamsMetaNames.MAX_NEW_TOKENS: 1024,
    #             GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
    #             GenTextParamsMetaNames.RANDOM_SEED: 42,
    #             GenTextParamsMetaNames.TEMPERATURE: 0,
    #         }
    #     }
    # },
    "Qwen/Qwen2.5-7B": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "openai_api_key": "/",
            "openai_api_base": f'http://{vllm_host}.pok.ibm.com:{vllm_port}/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
        }
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "openai_api_key": "/",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:8095/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            "default_headers": {},
        }
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        # MODEL_NAME="llama-3.1-70b-instruct"
        # KEY="pRt87ruBLVV443l"
        # PORT=7070
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "openai_api_key": "/",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:8007/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            "default_headers": {},
        }
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "openai_api_key": "/",
            "openai_api_base": f'http://0.0.0.0:8096/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "microsoft/phi-4": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "microsoft/phi-4",
            "openai_api_key": "/",
            "openai_api_base": f'http://0.0.0.0:8002/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "google/gemma-2-9b-it": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "google/gemma-2-9b-it",
            "openai_api_key": "/",
            "openai_api_base": f'http://0.0.0.0:8003/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "ibm-granite/granite-3.3-8b-instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ibm-granite/granite-3.3-8b-instruct",
            "openai_api_key": "/",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:8004/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "agentica-org/DeepCoder-14B-Preview": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "agentica-org/DeepCoder-14B-Preview",
            "openai_api_key": "/",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:8005/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "openai_api_key": "/",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:8006/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "llama-3.3-70b-instruct",
            "openai_api_key": "pRt87ruBLVV443l",
            # "openai_api_base": f'{rits_url}/llama-3-1-8b-instruct/v1',
            "openai_api_base": f'http://0.0.0.0:10001/v1',
            "max_tokens": 1024,
            # "temperature": 0,
            # "model_kwargs": {
            #     "stop": [""]
            # },
            # "default_headers": {'RITS_API_KEY': rits_api_key},
            "default_headers": {},
        }
    },
}
