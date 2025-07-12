#!/bin/bash
# Change working directory to the project root
cd "$(dirname "$0")"/.. || exit 1


uvicorn main:app --reload
