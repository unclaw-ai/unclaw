"""Static internal skill manifests for the shipped internal skills package."""

from __future__ import annotations

from unclaw.skills.models import (
    SkillAvailability,
    SkillBudgetMetadata,
    SkillInstallMode,
    SkillManifest,
    SkillPromptBudgetTier,
    SkillPromptFragment,
    SkillPromptFragmentKind,
    SkillPromptSource,
    SkillPromptSourceKind,
    SkillToolBinding,
)

_MODULE_REFERENCE = "unclaw.skills.manifests"


def _inline_source(fragment_id: str) -> SkillPromptSource:
    return SkillPromptSource(
        kind=SkillPromptSourceKind.INLINE,
        reference=f"{_MODULE_REFERENCE}:{fragment_id}",
    )


def _fragment(
    *,
    skill_id: str,
    fragment_suffix: str,
    name: str,
    kind: SkillPromptFragmentKind,
    description: str | None = None,
    lines: tuple[str, ...],
) -> SkillPromptFragment:
    fragment_id = f"{skill_id}.{fragment_suffix}"
    return SkillPromptFragment(
        fragment_id=fragment_id,
        skill_id=skill_id,
        name=name,
        kind=kind,
        source=_inline_source(fragment_id),
        lines=lines,
        description=description,
    )


_FABRICATION_THREE_D_PRINTER_SKILL = SkillManifest(
    skill_id="fabrication.three_d_printer",
    display_name="3D Printer Operations",
    version="0.1.0",
    description=(
        "Optional workflow skill for printer-state checks, print-job preparation, "
        "and hardware-facing safety guidance."
    ),
    install_mode=SkillInstallMode.OPT_IN,
    tags=("fabrication", "hardware", "printing"),
    prompt_fragments=(
        _fragment(
            skill_id="fabrication.three_d_printer",
            fragment_suffix="context.overview",
            name="3D printer overview",
            kind=SkillPromptFragmentKind.CONTEXT,
            lines=(
                "This skill scopes work around 3D printer status, queued jobs, and print preparation.",
            ),
        ),
        _fragment(
            skill_id="fabrication.three_d_printer",
            fragment_suffix="guidance.workflow",
            name="3D printer workflow guidance",
            kind=SkillPromptFragmentKind.GUIDANCE,
            lines=(
                "Keep nozzle, bed, material, and layer settings explicit when preparing print instructions.",
                "Prefer checking printer status before telling the user that a hardware action has started.",
            ),
        ),
        _fragment(
            skill_id="fabrication.three_d_printer",
            fragment_suffix="safety.hardware_state",
            name="3D printer hardware-state safety",
            kind=SkillPromptFragmentKind.SAFETY,
            lines=(
                "Do not claim a print started, stopped, or completed unless a printer skill tool returned that result in this conversation.",
            ),
        ),
    ),
    tool_bindings=(
        SkillToolBinding(
            binding_id="fabrication.three_d_printer.get_status",
            tool_name="printer_status",
            description="Retrieve current 3D printer state before hardware claims.",
            required=True,
        ),
        SkillToolBinding(
            binding_id="fabrication.three_d_printer.submit_job",
            tool_name="submit_print_job",
            description="Submit a prepared print job to the connected printer.",
        ),
    ),
    availability=SkillAvailability(
        supported_model_profiles=("main", "deep", "codex"),
        required_builtin_tool_names=("read_text_file",),
        requires_model_tool_support=True,
    ),
    budget=SkillBudgetMetadata(
        min_budget_tier=SkillPromptBudgetTier.STANDARD,
        prompt_priority=80,
        estimated_prompt_lines=4,
    ),
)


_MESSAGING_TELEGRAM_SKILL = SkillManifest(
    skill_id="messaging.telegram",
    display_name="Telegram Messaging",
    version="0.1.0",
    description=(
        "Optional workflow skill for Telegram-specific drafting, chat lookup, "
        "and delivery-side safety rules."
    ),
    install_mode=SkillInstallMode.OPT_IN,
    tags=("messaging", "telegram", "channels"),
    prompt_fragments=(
        _fragment(
            skill_id="messaging.telegram",
            fragment_suffix="context.overview",
            name="Telegram overview",
            kind=SkillPromptFragmentKind.CONTEXT,
            lines=(
                "This skill scopes work around Telegram chat context, message drafting, and delivery actions.",
            ),
        ),
        _fragment(
            skill_id="messaging.telegram",
            fragment_suffix="guidance.formatting",
            name="Telegram formatting guidance",
            kind=SkillPromptFragmentKind.GUIDANCE,
            lines=(
                "Keep channel posts concise and preserve URLs, mentions, and formatting-sensitive text unless the user asked to rewrite them.",
                "Separate draft generation from send actions so review can happen before delivery.",
            ),
        ),
        _fragment(
            skill_id="messaging.telegram",
            fragment_suffix="safety.delivery_state",
            name="Telegram delivery-state safety",
            kind=SkillPromptFragmentKind.SAFETY,
            lines=(
                "Do not claim a Telegram message was sent, edited, or deleted unless a Telegram skill tool returned that result in this conversation.",
            ),
        ),
    ),
    tool_bindings=(
        SkillToolBinding(
            binding_id="messaging.telegram.list_chats",
            tool_name="list_telegram_chats",
            description="List Telegram chats or channels available to the runtime.",
            required=True,
        ),
        SkillToolBinding(
            binding_id="messaging.telegram.send_message",
            tool_name="send_telegram_message",
            description="Send a prepared Telegram message to a selected chat.",
        ),
    ),
    availability=SkillAvailability(
        supported_model_profiles=("main", "deep", "codex"),
        requires_model_tool_support=True,
    ),
    budget=SkillBudgetMetadata(
        min_budget_tier=SkillPromptBudgetTier.COMPACT,
        prompt_priority=60,
        estimated_prompt_lines=4,
    ),
)


_INTERNAL_SKILL_MANIFESTS = (
    _FABRICATION_THREE_D_PRINTER_SKILL,
    _MESSAGING_TELEGRAM_SKILL,
)


def load_internal_skill_manifests() -> tuple[SkillManifest, ...]:
    """Return the shipped internal skill manifests in deterministic id order."""
    return tuple(sorted(_INTERNAL_SKILL_MANIFESTS, key=lambda skill: skill.skill_id))


__all__ = ["load_internal_skill_manifests"]
