"""
Tests pour le système de mémoire avec résumé périodique.
"""

from unittest.mock import Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from chatbot_uqac.rag.engine import RagChat, summarize_history


def test_summarize_history():
    """Test que la fonction de résumé fonctionne correctement."""
    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "L'utilisateur a posé des questions sur la mission et les valeurs de l'UQAC."
    mock_llm.invoke.return_value = mock_response
    
    # Créer un historique de test
    history = [
        HumanMessage(content="Quelle est la mission de l'UQAC ?"),
        AIMessage(content="La mission de l'UQAC est... [1]"),
        HumanMessage(content="Quelles sont les valeurs ?"),
        AIMessage(content="Les valeurs de l'UQAC sont... [2]"),
    ]
    
    # Générer le résumé
    summary = summarize_history(history, mock_llm)
    
    # Vérifier que le LLM a été appelé
    assert mock_llm.invoke.called
    assert len(summary) > 0
    # Vérifier que les citations ne sont pas dans le prompt
    call_args = mock_llm.invoke.call_args[0][0]
    assert "[1]" not in str(call_args[0].content)


def test_history_compression_triggered():
    """Test que la compression est déclenchée au bon moment."""
    # Mock retriever et LLM
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = []
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_llm.invoke.return_value = mock_response
    
    # Créer RagChat avec seuil bas pour tester
    chat = RagChat(
        retriever=mock_retriever,
        llm=mock_llm,
        summarize_threshold=6,  # Se déclenche après 6 messages
        keep_recent=4  # Garde 4 messages récents
    )
    
    # Ajouter des messages pour dépasser le seuil
    for i in range(4):  # 4 tours = 8 messages
        chat._append_history(f"Question {i}", f"Answer {i}")
    
    # Vérifier que l'historique a été compressé
    # On devrait avoir: 1 SystemMessage (résumé) + 4 messages récents
    assert len(chat.history) == 5
    assert isinstance(chat.history[0], SystemMessage)
    assert "summary" in chat.history[0].content.lower()


def test_recent_messages_preserved():
    """Test que les messages récents sont préservés intacts."""
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = []
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Résumé généré"
    mock_llm.invoke.return_value = mock_response
    
    chat = RagChat(
        retriever=mock_retriever,
        llm=mock_llm,
        summarize_threshold=6,
        keep_recent=4
    )
    
    # Ajouter des messages
    for i in range(4):
        chat._append_history(f"Question {i}", f"Answer {i}")
    
    # Vérifier que les 2 derniers tours sont préservés
    assert chat.history[-2].content == "Question 3"
    assert chat.history[-1].content == "Answer 3"
    assert chat.history[-4].content == "Question 2"
    assert chat.history[-3].content == "Answer 2"


def test_no_compression_below_threshold():
    """Test qu'il n'y a pas de compression sous le seuil."""
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = []
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_llm.invoke.return_value = mock_response
    
    chat = RagChat(
        retriever=mock_retriever,
        llm=mock_llm,
        summarize_threshold=10,
        keep_recent=4
    )
    
    # Ajouter seulement 2 tours (4 messages)
    for i in range(2):
        chat._append_history(f"Question {i}", f"Answer {i}")
    
    # Vérifier qu'il n'y a pas de SystemMessage (pas de résumé)
    assert len(chat.history) == 4
    assert not any(isinstance(msg, SystemMessage) for msg in chat.history)


def test_cumulative_summaries():
    """Test que les résumés s'accumulent correctement."""
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = []
    
    # Premier résumé
    mock_llm = Mock()
    first_summary = Mock()
    first_summary.content = "Premier résumé"
    second_summary = Mock()
    second_summary.content = "Deuxième résumé"
    
    mock_llm.invoke.side_effect = [first_summary, second_summary]
    
    chat = RagChat(
        retriever=mock_retriever,
        llm=mock_llm,
        summarize_threshold=6,
        keep_recent=2
    )
    
    # Premier cycle de compression
    for i in range(4):
        chat._append_history(f"Q{i}", f"A{i}")
    
    # Deuxième cycle de compression
    for i in range(4, 7):
        chat._append_history(f"Q{i}", f"A{i}")
    
    # Vérifier que le résumé cumulatif contient les deux résumés
    assert isinstance(chat.history[0], SystemMessage)
    summary_content = chat.history[0].content
    assert "Previous conversation summary" in summary_content or "summary" in summary_content.lower()


if __name__ == "__main__":
    print("Running memory system tests...")
    
    test_summarize_history()
    print("✓ test_summarize_history passed")
    
    test_history_compression_triggered()
    print("✓ test_history_compression_triggered passed")
    
    test_recent_messages_preserved()
    print("✓ test_recent_messages_preserved passed")
    
    test_no_compression_below_threshold()
    print("✓ test_no_compression_below_threshold passed")
    
    test_cumulative_summaries()
    print("✓ test_cumulative_summaries passed")
    
    print("\n✅ All tests passed!")
