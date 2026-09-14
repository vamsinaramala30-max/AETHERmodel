"""
AETHER MODEL — Context Window Manager
Formats prompt history and structured context into standard tokenized boundaries:
SYSTEM -> MEMORY -> RAG EVIDENCE -> PROJECT CONTEXT -> TOOL RESULTS -> CONVERSATION HISTORY -> USER REQUEST -> ASSISTANT
"""

from typing import Dict, Any, List, Optional
from tokenizer.tokenizer import AetherTokenizer


def _tok_id(tokenizer: AetherTokenizer, name: str, fallback: int) -> int:
    """Resolve a special-token ID from the LOADED tokenizer.

    Never use hard-coded constants here — the legacy tokenizer.py constants
    (USER=4, ASST=5, SYS=3) differ from the BPE bpe.py constants
    (USER=5, ASST=6, SYS=4), causing corrupted prompt structure.
    """
    return tokenizer.token_to_id.get(name, fallback)

class ContextManager:
    def __init__(self, tokenizer: AetherTokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def build_audit_representation(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Builds a safe, structured audit representation showing active context segments.
        Used for observability, debugging, and testing without exposing raw tokens.
        """
        context = context or {}
        return {
            "has_system_prompt": bool(context.get("system_prompt") or context.get("systemPrompt")),
            "has_memory_context": bool(context.get("memory_context") or context.get("memoryContext")),
            "has_rag_context": bool(context.get("rag_context") or context.get("ragContext")),
            "has_project_context": bool(context.get("project_context") or context.get("projectContext")),
            "has_tool_results": bool(context.get("tool_results") or context.get("toolResults")),
            "conversation_turns": len(context.get("conversation_history") or context.get("conversationHistory") or context.get("messages") or []),
            "prompt_length_chars": len(prompt),
            "max_seq_len": self.max_seq_len,
            "generation_settings": {
                "max_tokens": context.get("max_tokens", 256),
                "temperature": context.get("temperature", 0.7),
                "top_k": context.get("top_k", 40),
                "top_p": context.get("top_p", 0.9),
                "repetition_penalty": context.get("repetition_penalty", 1.15),
            }
        }

    def format_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Formats raw prompt string and contextual data into structured token sequence.
        Context Sequence:
        SYSTEM -> MEMORY -> RAG EVIDENCE -> PROJECT CONTEXT -> TOOL RESULTS -> CONVERSATION HISTORY -> USER REQUEST -> ASSISTANT

        Applies intelligent prioritization:
        Current User Prompt & Assistant Prefix (High Priority) >
        Critical System Instructions >
        Recent Conversation History >
        Approved Memory >
        RAG / Evidence >
        Older History
        """
        context = context or {}

        # Resolve role token IDs from the live tokenizer (not legacy constants)
        USER_ID = _tok_id(self.tokenizer, "<user>", 4)
        ASST_ID = _tok_id(self.tokenizer, "<assistant>", 5)
        SYS_ID  = _tok_id(self.tokenizer, "<system>", 3)
        TOOL_ID = _tok_id(self.tokenizer, "<tool>", 6)
        EVI_ID  = _tok_id(self.tokenizer, "<evidence>", 7)

        # 1. Essential tokens (Priority 1)
        user_ids = [USER_ID] + self.tokenizer.encode(prompt)
        asst_prefix = [ASST_ID]
        essential_len = len(user_ids) + len(asst_prefix)

        # Budget remaining for optional context
        available_budget = max(0, self.max_seq_len - essential_len - 1)

        # 2. System Instructions (Priority 2)
        system_ids: List[int] = []
        system_prompt = context.get("system_prompt") or context.get("systemPrompt")
        if system_prompt:
            enc_sys = self.tokenizer.encode(str(system_prompt))
            system_ids = [SYS_ID] + enc_sys

        # 3. Conversation History (Priority 3 - recent turns prioritized)
        history_ids: List[int] = []
        conversation_history = context.get("conversation_history") or context.get("conversationHistory") or context.get("messages")
        if conversation_history and isinstance(conversation_history, list):
            # Encode each message backwards (most recent first)
            turn_token_blocks: List[List[int]] = []
            for msg in reversed(conversation_history):
                role = msg.get("role", "user") if isinstance(msg, dict) else "user"
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                if not content:
                    continue
                role_tok = SYS_ID if role in ("system", "SYSTEM") else (
                    ASST_ID if role in ("assistant", "ASSISTANT") else USER_ID
                )
                block = [role_tok] + self.tokenizer.encode(content)
                turn_token_blocks.append(block)

            # Assemble forward
            for block in reversed(turn_token_blocks):
                history_ids.extend(block)

        # 4. Memory context (Priority 4)
        memory_ids: List[int] = []
        memory_context = context.get("memory_context") or context.get("memoryContext")
        if memory_context:
            memory_ids = [SYS_ID] + self.tokenizer.encode("[Memory]: {}".format(memory_context))

        # 5. RAG / Evidence context (Priority 5)
        rag_ids: List[int] = []
        rag_context = context.get("rag_context") or context.get("ragContext")
        if rag_context:
            rag_ids = [EVI_ID] + self.tokenizer.encode(str(rag_context))

        # 6. Project & Tool context (Priority 6)
        project_ids: List[int] = []
        project_context = context.get("project_context") or context.get("projectContext")
        if project_context:
            project_ids = [SYS_ID] + self.tokenizer.encode("[Project]: {}".format(project_context))

        tool_ids: List[int] = []
        tool_results = context.get("tool_results") or context.get("toolResults")
        if tool_results:
            tool_ids = [TOOL_ID] + self.tokenizer.encode(str(tool_results))

        # Build candidate contextual payload in structured sequence
        context_parts: List[int] = []
        context_parts.extend(system_ids)
        context_parts.extend(memory_ids)
        context_parts.extend(rag_ids)
        context_parts.extend(project_ids)
        context_parts.extend(tool_ids)
        context_parts.extend(history_ids)

        # Apply trimming if context parts exceed available budget
        if len(context_parts) > available_budget:
            # Trim from older/lower-priority components while keeping system & recent tokens
            if len(system_ids) <= available_budget:
                remaining_after_sys = available_budget - len(system_ids)
                non_sys_parts = context_parts[len(system_ids):]
                trimmed_non_sys = non_sys_parts[-remaining_after_sys:] if remaining_after_sys > 0 else []
                context_parts = system_ids + trimmed_non_sys
            else:
                context_parts = context_parts[-available_budget:] if available_budget > 0 else []

        full_token_ids = context_parts + user_ids + asst_prefix

        # Final invariant check: never exceed max_seq_len - 1
        if len(full_token_ids) >= self.max_seq_len:
            full_token_ids = full_token_ids[-(self.max_seq_len - 1):]

        return full_token_ids

