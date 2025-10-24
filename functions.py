import os
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from collections import Counter


def is_finance_query(query: str, threshold=0.65):
    res = domain_classifier(query, ALLOWED_TOPICS)
    return res["scores"][0] >= threshold

