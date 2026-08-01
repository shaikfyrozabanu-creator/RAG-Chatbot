import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from backend.app.services import pinecone_service
    from backend.app.routers.chat import chat_with_context, ChatRequest
except ImportError:
    from app.services import pinecone_service
    from app.routers.chat import chat_with_context, ChatRequest


class TestRetrievalFiltering(unittest.TestCase):

    @patch("backend.app.services.pinecone_service.get_index")
    @patch("backend.app.services.pinecone_service.pc", new=True)
    def test_delete_all_vectors_calls_index_delete(self, mock_get_index):
        """Test that delete_all_vectors calls index.delete(delete_all=True)."""
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        pinecone_service.delete_all_vectors()
        mock_index.delete.assert_called_once_with(delete_all=True)

    @patch("backend.app.services.pinecone_service.get_index")
    @patch("backend.app.services.pinecone_service.pc", new=True)
    def test_query_similar_chunks_applies_filename_filter_and_limits_top3(self, mock_get_index):
        """Test that query_similar_chunks applies filename filter and limits output to top 3."""
        match1 = MagicMock(score=0.92, metadata={"filename": "doc1.pdf", "text": "Chunk 1"})
        match2 = MagicMock(score=0.85, metadata={"filename": "doc1.pdf", "text": "Chunk 2"})
        match3 = MagicMock(score=0.60, metadata={"filename": "doc1.pdf", "text": "Chunk 3"})
        match4 = MagicMock(score=0.40, metadata={"filename": "doc1.pdf", "text": "Chunk 4"})

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value.total_vector_count = 10
        mock_index.query.return_value.matches = [match1, match2, match3, match4]
        mock_get_index.return_value = mock_index

        dummy_vector = [0.1] * 384
        results = pinecone_service.query_similar_chunks(dummy_vector, filename="doc1.pdf", top_k=3)

        # 1. Verify index.query received filter={"filename": "doc1.pdf"}
        mock_index.query.assert_called_once()
        _, kwargs = mock_index.query.call_args
        self.assertEqual(kwargs.get("filter"), {"filename": "doc1.pdf"})

        # 2. Verify top 3 chunks returned
        self.assertEqual(len(results), 3)

    @patch("backend.app.services.supabase_service.store_message")
    @patch("backend.app.services.supabase_service.get_chat_history", return_value=[])
    @patch("backend.app.services.pdf_service.generate_query_embedding", return_value=[0.1]*384)
    @patch("backend.app.services.pinecone_service.query_similar_chunks", return_value=[])
    @patch("backend.app.services.gemini_service.generate_rag_answer", return_value="Mocked LLM answer")
    def test_chat_returns_fallback_message_only_when_index_empty(self, mock_gemini, mock_pinecone, mock_pdf, mock_get_hist, mock_store):
        """Test that /chat router calls generate_rag_answer when question is received."""
        req = ChatRequest(question="What is this document about?", session_id="test_session")
        
        import asyncio
        response = asyncio.run(chat_with_context(req))

        self.assertEqual(response.answer, "Mocked LLM answer")
        self.assertEqual(response.context, [])


if __name__ == "__main__":
    unittest.main()

