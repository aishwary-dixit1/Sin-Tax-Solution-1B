# Persona-Driven Document Intelligence (Round 1B)

This solution is an intelligent document analyst that extracts and ranks the most relevant sections from a collection of PDFs based on a specified persona and their job-to-be-done.

## Overview

The pipeline is designed to be highly generalizable across different domains (e.g., travel, legal, food). It uses an advanced, feature-based document parser to structure the PDFs and a powerful semantic model (`all-mpnet-base-v2`) to understand the context of the user's query and rank the content. The final output provides a diverse and highly relevant set of sections to help the user accomplish their task.

## How to Run (Docker)

This solution is packaged in a Docker container for easy and consistent execution.

### Build the Docker Image

From the root of the repository, run the following command to build the image:
```bash
docker build --platform linux/amd64 -t your-solution-name .