#### **`approach_explanation.md`**
This file provides the detailed, 300-500 word explanation of your methodology as required by the problem statement.

```markdown
# Approach Explanation: Persona-Driven Document Intelligence

Our solution is a multi-stage pipeline designed to function as a generalizable and intelligent document analyst. The core challenge is to semantically understand a user's abstract goal—defined by a persona and a job-to-be-done—and extract the most relevant, actionable sections from a diverse collection of PDF documents. Our approach prioritizes robust parsing, deep semantic understanding, and an intelligent ranking strategy to deliver accurate results across any domain.

### 1. Advanced Document Structuring

The pipeline begins by structuring the raw PDF content. Instead of relying on a resource-heavy secondary model, we've implemented an advanced, feature-based parser using the PyMuPDF library. For each line of text, this parser engineers a set of features, including relative font size, font weight (boldness), text length, and structural cues (e.g., presence of bullets or ending punctuation). A scoring heuristic then classifies each line as a heading or body text. This method is extremely fast, has no model overhead, and accurately identifies section boundaries, which is critical for creating coherent content chunks.

### 2. Semantic Representation

To understand the context of both the user's query and the document content, we use the powerful `all-MiniLM-L6-v2` sentence-transformer model. A dynamic query is generated from the input persona and job description, creating a rich contextual prompt for the model. Each parsed section from the documents is then converted into a high-fidelity vector embedding. This allows for a nuanced, meaning-based comparison rather than simple keyword matching.

### 3. Hybrid Ranking for Relevance and Diversity

The core of our ranking logic is a hybrid strategy that balances pure relevance with a diverse output. First, cosine similarity is used to score every document chunk against the query embedding. Generic, boilerplate sections like "Introduction" or "Conclusion" are filtered out. From the remaining high-scoring chunks, we create a candidate pool of the top 20 most relevant sections. Finally, to ensure the output is comprehensive, we select the single best-scoring section from each unique document within this high-quality pool. This hybrid approach prevents the results from being dominated by a single, highly relevant document and provides the user with a balanced and actionable overview, fulfilling the role of a trusted research companion.