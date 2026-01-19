"""
Démonstration du système de mémoire avec résumé périodique.

Ce script montre comment le système de résumé fonctionne en pratique.
"""

from unittest.mock import Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from chatbot_uqac.rag.engine import RagChat, summarize_history


def demo_basic_summarization():
    """Démo basique : voir comment le résumé se déclenche."""
    print("=" * 70)
    print("DÉMO 1 : Déclenchement automatique du résumé")
    print("=" * 70)
    
    # Mock retriever et LLM
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = [
        Mock(page_content="Contenu exemple", metadata={"url": "https://example.com"})
    ]
    
    mock_llm = Mock()
    # Simuler les réponses du LLM
    responses = [
        Mock(content="Réponse 1"),
        Mock(content="Réponse 2"),
        Mock(content="Réponse 3"),
        Mock(content="Réponse 4"),
        Mock(content="Résumé: L'utilisateur a posé des questions sur l'UQAC"),
    ]
    mock_llm.invoke.side_effect = responses
    
    # Créer le chat avec seuil bas pour la démo
    chat = RagChat(
        retriever=mock_retriever,
        llm=mock_llm,
        summarize_threshold=6,  # Se déclenche après 6 messages
        keep_recent=4  # Garde 4 messages récents
    )
    
    # Simuler une conversation
    questions = [
        "Quelle est la mission de l'UQAC ?",
        "Quelles sont les valeurs ?",
        "Parle-moi de la planification",
        "Comment fonctionne l'admission ?",
    ]
    
    for i, question in enumerate(questions):
        print(f"\n--- Tour {i+1} ---")
        print(f"Q: {question}")
        
        # Ajouter manuellement pour la démo
        chat._append_history(question, f"Réponse {i+1}")
        
        print(f"Nombre de messages dans l'historique: {len(chat.history)}")
        
        # Afficher le type du premier message si c'est un résumé
        if chat.history and isinstance(chat.history[0], SystemMessage):
            print(f"✨ RÉSUMÉ DÉTECTÉ: {chat.history[0].content[:60]}...")
        
    print("\n" + "=" * 70)
    print("RÉSULTAT FINAL:")
    print("=" * 70)
    print(f"Total des messages: {len(chat.history)}")
    print(f"Types: ", end="")
    for msg in chat.history:
        if isinstance(msg, SystemMessage):
            print("📋", end="")
        elif isinstance(msg, HumanMessage):
            print("👤", end="")
        elif isinstance(msg, AIMessage):
            print("🤖", end="")
    print()


def demo_history_structure():
    """Démo : structure de l'historique avant et après résumé."""
    print("\n\n" + "=" * 70)
    print("DÉMO 2 : Structure de l'historique")
    print("=" * 70)
    
    # Créer un historique exemple
    history_before = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Réponse 1 avec citation [1]"),
        HumanMessage(content="Question 2"),
        AIMessage(content="Réponse 2 avec citation [2]"),
        HumanMessage(content="Question 3"),
        AIMessage(content="Réponse 3 avec citation [3]"),
    ]
    
    print("\n📝 AVANT RÉSUMÉ:")
    print("-" * 70)
    for i, msg in enumerate(history_before):
        msg_type = type(msg).__name__
        content = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
        print(f"  [{i}] {msg_type}: {content}")
    
    print(f"\nTaille totale: {len(history_before)} messages")
    
    # Simuler un résumé
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(
        content="L'utilisateur a posé 3 questions sur l'UQAC et reçu des réponses détaillées."
    )
    
    summary = summarize_history(history_before[:4], mock_llm)
    
    history_after = [
        SystemMessage(content=f"Conversation summary: {summary}"),
        HumanMessage(content="Question 3"),
        AIMessage(content="Réponse 3 avec citation [3]"),
    ]
    
    print("\n📋 APRÈS RÉSUMÉ:")
    print("-" * 70)
    for i, msg in enumerate(history_after):
        msg_type = type(msg).__name__
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  [{i}] {msg_type}: {content}")
    
    print(f"\nTaille totale: {len(history_after)} messages")
    print(f"Réduction: {len(history_before)} → {len(history_after)} messages "
          f"({100 - int(len(history_after)/len(history_before)*100)}% de réduction)")


def demo_cumulative_summaries():
    """Démo : résumés cumulatifs."""
    print("\n\n" + "=" * 70)
    print("DÉMO 3 : Résumés cumulatifs")
    print("=" * 70)
    
    print("\n📚 Scénario: Longue conversation nécessitant plusieurs résumés")
    
    print("\n1️⃣ Premier cycle (messages 1-6):")
    print("   → Résumé 1: 'L'utilisateur s'est renseigné sur la mission et les valeurs'")
    
    print("\n2️⃣ Messages récents 7-10 conservés intacts")
    
    print("\n3️⃣ Nouveaux messages 11-14 ajoutés")
    print("   → Besoin d'un deuxième résumé!")
    
    print("\n4️⃣ Résumé cumulatif créé:")
    print("   📋 SystemMessage:")
    print("      'Previous conversation summary: L'utilisateur s'est renseigné")
    print("       sur la mission et les valeurs.")
    print("       Recent topics: Discussion sur l'admission et les programmes.'")
    
    print("\n5️⃣ Structure finale:")
    print("   [Résumé cumulatif] + [Messages 13-14]")
    
    print("\n✅ Avantage: Contexte complet préservé avec une taille minimale!")


def demo_citation_removal():
    """Démo : suppression des citations dans les résumés."""
    print("\n\n" + "=" * 70)
    print("DÉMO 4 : Suppression des citations")
    print("=" * 70)
    
    print("\n🎯 Objectif: Les résumés ne doivent pas contenir de citations")
    
    original = "La mission de l'UQAC est définie [1] et inclut l'enseignement [2]."
    print(f"\n📄 Réponse originale:")
    print(f"   '{original}'")
    
    import re
    cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", original)
    print(f"\n🧹 Après nettoyage:")
    print(f"   '{cleaned}'")
    
    print(f"\n✅ Résultat: Citations supprimées, texte préservé")
    print(f"   Pourquoi? Les citations dans un résumé seraient trompeuses")


def demo_performance_impact():
    """Démo : impact sur les performances."""
    print("\n\n" + "=" * 70)
    print("DÉMO 5 : Impact sur les performances")
    print("=" * 70)
    
    print("\n📊 Simulation de taille de prompts:")
    
    # Simuler des tailles de messages
    avg_question_tokens = 20
    avg_answer_tokens = 200
    avg_context_tokens = 1000
    summary_tokens = 50
    
    # Sans résumé (20 messages)
    messages_without = 20
    tokens_without = (
        avg_context_tokens +  # Contexte RAG
        messages_without * (avg_question_tokens + avg_answer_tokens)
    )
    
    # Avec résumé (1 résumé + 6 messages récents)
    messages_with = 6
    tokens_with = (
        avg_context_tokens +  # Contexte RAG
        summary_tokens +  # Résumé
        messages_with * (avg_question_tokens + avg_answer_tokens)
    )
    
    print(f"\n🔴 SANS résumé (20 messages):")
    print(f"   Tokens: {tokens_without}")
    print(f"   Temps estimé: ~15 secondes")
    
    print(f"\n🟢 AVEC résumé (résumé + 6 messages):")
    print(f"   Tokens: {tokens_with}")
    print(f"   Temps estimé: ~5 secondes")
    
    reduction = 100 - int(tokens_with / tokens_without * 100)
    speedup = tokens_without / tokens_with
    
    print(f"\n📈 Amélioration:")
    print(f"   Réduction de tokens: {reduction}%")
    print(f"   Accélération: {speedup:.1f}x plus rapide")
    print(f"   Économie: {tokens_without - tokens_with} tokens par requête")


def main():
    """Exécuter toutes les démos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "DÉMONSTRATION DU SYSTÈME DE MÉMOIRE" + " " * 23 + "║")
    print("║" + " " * 15 + "avec Résumé Périodique" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        demo_basic_summarization()
        demo_history_structure()
        demo_cumulative_summaries()
        demo_citation_removal()
        demo_performance_impact()
        
        print("\n\n" + "=" * 70)
        print("🎉 FIN DES DÉMONSTRATIONS")
        print("=" * 70)
        print("\n✨ Le système de résumé est maintenant prêt à l'emploi!")
        print("📚 Consultez docs/MEMORY_SYSTEM.md pour plus de détails")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la démo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
