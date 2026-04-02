"""
Core DocuBot class responsible for:
  - Loading documents from the docs/ folder
  - Building a simple retrieval index (Phase 1)
  - Retrieving relevant snippets (Phase 1)
  - Supporting retrieval only answers
  - Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob


class DocuBot:

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()   # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the
        documents they appear in.

        Structure:
            {
                "token":    ["AUTH.md", "API_REFERENCE.md"],
                "database": ["DATABASE.md"],
            }

        Splits on whitespace, lowercases tokens, strips common
        punctuation, and records each filename only once per word.
        """
        index = {}
        for filename, text in documents:
            words = text.lower().split()
            seen_in_doc = set()
            for word in words:
                # Strip surrounding punctuation
                word = word.strip('.,!?;:"\'()[]{}<>')
                if not word or word in seen_in_doc:
                    continue
                seen_in_doc.add(word)
                if word not in index:
                    index[word] = []
                index[word].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches
        the query.

        Strategy:
          - Convert query into lowercase words (strip punctuation).
          - Count how many unique query words appear anywhere in the
            lowercased document text.
          - Return that count as the score.
        """
        text_lower = text.lower()
        query_words = query.lower().split()
        score = 0
        seen = set()
        for word in query_words:
            word = word.strip('.,!?;:"\'()[]{}<>')
            if word and word not in seen:
                seen.add(word)
                if word in text_lower:
                    score += 1
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant
        document snippets.

        Steps:
          1. Use the inverted index to find candidate documents that
             contain at least one query word.
          2. Score each candidate with score_document().
          3. Keep only documents with score > 0 (guardrail: return
             nothing if nothing is relevant).
          4. Return top_k results sorted by score descending as a
             list of (filename, text) tuples.
        """
        # Step 1: find candidate filenames via the index
        query_words = query.lower().split()
        candidate_files = set()
        for word in query_words:
            word = word.strip('.,!?;:"\'()[]{}<>')
            if word in self.index:
                for filename in self.index[word]:
                    candidate_files.add(filename)

        # Step 2 & 3: score candidates, filter out score == 0
        scored = []
        doc_map = {filename: text for filename, text in self.documents}
        for filename in candidate_files:
            text = doc_map.get(filename, "")
            score = self.score_document(query, text)
            if score > 0:
                scored.append((score, filename, text))

        # Step 4: sort descending and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(filename, text) for _, filename, text in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)
        if not snippets:
            return "I do not know based on these docs."
        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")
        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )
        snippets = self.retrieve(query, top_k=top_k)
        if not snippets:
            return "I do not know based on these docs."
        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
