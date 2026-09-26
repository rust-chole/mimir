"""Golden regression scenarios (test data only - nothing here is a production rule).

Each scenario is a synthetic VOD with real TTS speech, real face imagery,
known ground truth (words, speakers, events) and a scripted editor that stands
in for the model roles, so the full production pipeline can run end to end
with real FFmpeg/OpenCV processing and real renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tests.synth.scene import PersonPlacement


@dataclass
class Speaker:
    voice: str
    speed: int = 165
    pitch: int = 50
    placement: PersonPlacement | None = None


@dataclass
class Moment:
    anchor: str                     # "line:<index>" | "event:<index>" | "lines:<a>-<b>"
    type: str
    strength: float
    label: str
    before: float = 5.0
    after: float = 2.5
    source: str = "transcript"


@dataclass
class Scenario:
    name: str
    kind: str                       # talk | gameplay | irl | ui
    duration: float
    speakers: dict[str, Speaker]
    lines: list[tuple[str, float, str]]
    events: list[tuple[float, str]] = field(default_factory=list)
    moments: list[Moment] = field(default_factory=list)
    story: dict[str, Any] = field(default_factory=dict)
    visual_events: list[dict[str, Any]] = field(default_factory=list)
    observer: Callable[[float], list[dict[str, Any]]] | None = None
    director: dict[str, list[str]] = field(default_factory=dict)
    hooks: list[str] = field(default_factory=list)
    accent: dict[str, Any] | None = None
    asr: dict[str, dict[str, str]] = field(default_factory=dict)   # role/view -> {word: heard}
    entities: tuple[str, ...] = ()
    speaker_names: tuple[tuple[str, str], ...] = ()
    expect: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_DIRECTOR = {
    "cold_open": ["REACTION", "ACTION_REGION", "SPEAKER_PUNCH", "TWO_SHOT", "GAMEPLAY_PRIORITY", "SCREEN_PRIORITY",
                  "WIDE_CONTEXT"],
    "setup": ["SCREEN_PRIORITY", "GAMEPLAY_PRIORITY", "SPEAKER_MEDIUM", "TWO_SHOT", "WIDE_CONTEXT"],
    "escalation": ["SCREEN_PRIORITY", "GAMEPLAY_PRIORITY", "SPEAKER_MEDIUM", "TWO_SHOT", "WIDE_CONTEXT"],
    "payoff": ["ACTION_REGION", "SCREEN_PRIORITY", "GAMEPLAY_PRIORITY", "SPEAKER_PUNCH", "REACTION", "TWO_SHOT",
               "WIDE_CONTEXT"],
    "reaction": ["REACTION", "SCREEN_PRIORITY", "GAMEPLAY_PRIORITY", "TWO_SHOT", "SPEAKER_MEDIUM", "WIDE_CONTEXT"],
    "aftermath": ["HOLD", "WIDE_CONTEXT"],
}


def single_speaker() -> Scenario:
    return Scenario(
        name="single_speaker", kind="talk", duration=58.0,
        speakers={"A": Speaker("en-us+m3", 160, 45, PersonPlacement(0, 0.46, 0.14, 0.66, phase=0.4))},
        lines=[
            ("A", 1.0, "Okay chat, welcome back to the stream."),
            ("A", 5.2, "Let me check the donations real quick."),
            ("A", 9.8, "Somebody bet me I could not finish the ghost pepper wings."),
            ("A", 15.6, "So I ordered the hottest ones on the menu."),
            ("A", 20.4, "I ate the first wing and honestly it was fine."),
            ("A", 25.0, "Then the second one hit different."),
            ("A", 30.6, "Wait, my face is actually melting right now!"),
            ("A", 36.2, "Chat, I cannot feel my tongue anymore."),
            ("A", 41.0, "Anyway, that is it for today."),
            ("A", 45.0, "Thanks for watching and stay hydrated."),
        ],
        events=[(33.4, "scream"), (37.9, "cheer")],
        moments=[Moment("line:6", "reaction", 8.6, "his face is melting from the ghost pepper", 12.0, 3.5),
                 Moment("line:7", "punchline", 7.2, "cannot feel his tongue", 3.0, 2.0)],
        story={"start_line": 2, "payoff": "line:6", "end_line": 7, "title": "Ghost pepper wings melt his face",
               "hook": "ghost pepper challenge", "emotion": "funny", "highlights": ["melting"]},
        hooks=["His face started melting", "The wings fought back"],
        accent={"line": 7, "word": 2, "category": "disbelief"},
        expect={"layout": "talking_head", "min_duration": 18.0},
    )


def two_speakers() -> Scenario:
    return Scenario(
        name="two_speakers", kind="talk", duration=56.0,
        speakers={"A": Speaker("en-us+m3", 165, 40, PersonPlacement(0, 0.29, 0.16, 0.62, phase=0.3)),
                  "B": Speaker("en-us+f2", 170, 70, PersonPlacement(1, 0.71, 0.18, 0.60, phase=1.7))},
        lines=[
            ("A", 1.0, "So how was the trip to Japan?"),
            ("B", 4.2, "It was amazing, the food was incredible."),
            ("A", 8.6, "Did you try the famous sushi place?"),
            ("B", 12.4, "Yes, and I ordered the chef special."),
            ("A", 16.4, "What was in it?"),
            ("B", 18.6, "I still do not know, it was moving."),
            ("A", 23.0, "Wait, the food was still moving?"),
            ("B", 26.6, "Yes, and I ate it anyway."),
            ("A", 30.0, "No way, you are crazy!"),
            ("B", 34.4, "It tasted like the ocean."),
            ("A", 38.4, "Okay, next topic please."),
        ],
        events=[(31.2, "cheer")],
        moments=[Moment("line:7", "unexpected_answer", 8.4, "she ate the moving dish anyway", 10.0, 4.0),
                 Moment("line:5", "reveal", 7.0, "the dish was moving", 4.0, 2.0)],
        story={"start_line": 2, "payoff": "line:7", "end_line": 9, "title": "She ate the moving sushi",
               "hook": "the chef special", "emotion": "shock", "highlights": ["moving"]},
        hooks=["The chef special was alive", "She ate it anyway"],
        expect={"layout": "multi_person", "two_shot_in": ["payoff", "reaction"]},
    )


def multi_interruption() -> Scenario:
    return Scenario(
        name="multi_interruption", kind="talk", duration=58.0,
        speakers={"A": Speaker("en-us+m3", 165, 40, PersonPlacement(0, 0.20, 0.22, 0.50, phase=0.2)),
                  "B": Speaker("en-us+f2", 172, 72, PersonPlacement(1, 0.50, 0.20, 0.52, phase=1.1)),
                  "C": Speaker("en-gb+m1", 160, 35, PersonPlacement(2, 0.80, 0.23, 0.50, phase=2.3))},
        lines=[
            ("A", 1.0, "Alright everyone, the vote is in."),
            ("B", 4.6, "Finally, I have been waiting all day."),
            ("C", 8.4, "Just read the results already."),
            ("A", 12.0, "The winner of the cooking contest is"),
            ("B", 15.3, "Please say me, please say me!"),
            ("A", 16.2, "the one who burned the kitchen."),
            ("C", 20.2, "That was me, I burned the kitchen!"),
            ("B", 21.4, "No, that is not fair at all!"),
            ("A", 25.2, "The judges loved the smoky flavor."),
            ("C", 29.0, "Smoky flavor, I call it talent."),
            ("B", 32.6, "I am never cooking with you again."),
            ("A", 37.0, "Okay, that is the end of the show."),
        ],
        events=[(23.8, "cheer")],
        moments=[Moment("lines:5-7", "reversal", 8.7, "the kitchen burner wins the contest", 9.0, 5.0),
                 Moment("line:9", "punchline", 7.4, "I call it talent", 3.0, 2.0)],
        story={"start_line": 3, "payoff": "line:5", "end_line": 10, "title": "The kitchen burner won",
               "hook": "the cooking contest", "emotion": "funny", "highlights": ["burned"]},
        hooks=["The worst cook won", "They crowned the arsonist"],
        expect={"overlap_lane": True},
    )


def gameplay_facecam() -> Scenario:
    return Scenario(
        name="gameplay_facecam", kind="gameplay", duration=56.0,
        speakers={"A": Speaker("en-us+m3", 175, 50, PersonPlacement(0, 0.885, 0.70, 0.27, phase=0.5, sway=0.3))},
        lines=[
            ("A", 1.5, "Okay we are in the final round now."),
            ("A", 6.0, "I only have one life left, chat."),
            ("A", 10.6, "Watch me clutch this whole lobby."),
            ("A", 15.4, "They are all coming from the left side."),
            ("A", 20.2, "I am throwing the big one right now."),
            ("A", 27.0, "Oh my god, it wiped the entire team!"),
            ("A", 32.4, "Did you see that, chat, did you see that?"),
            ("A", 38.0, "Okay, back to the lobby."),
        ],
        events=[(24.6, "impact"), (25.0, "scream")],
        moments=[Moment("event:0", "visual_impact", 9.1, "explosion wipes the enemy team", 10.0, 4.0, "both"),
                 Moment("line:6", "reaction", 7.5, "did you see that", 3.0, 2.0)],
        story={"start_line": 1, "payoff": "event:0", "end_line": 6, "title": "One grenade wiped the lobby",
               "hook": "last life clutch", "emotion": "hype", "highlights": ["wiped"]},
        visual_events=[{"at": "event:0", "type": "explosion", "description": "large explosion flash in the center of "
                                                                          "the gameplay; enemies disappear",
                        "confidence": 0.9}],
        hooks=["Last life, whole lobby", "One throw ended it"],
        extra={"explosion_event": 0},
        expect={"layout": "facecam_gameplay", "stack": True},
    )


def irl_object() -> Scenario:
    return Scenario(
        name="irl_object", kind="irl", duration=56.0,
        speakers={"A": Speaker("en-us+m3", 165, 45, PersonPlacement(0, 0.24, 0.20, 0.55, phase=0.8))},
        lines=[
            ("A", 1.0, "Today we are building the tallest box tower ever."),
            ("A", 6.4, "I stacked five boxes this morning."),
            ("A", 11.0, "Now I just need to add one more on top."),
            ("A", 16.2, "Let me walk over there carefully."),
            ("A", 20.8, "Okay, reaching up, almost there."),
            ("A", 28.2, "Oh no, all my work is gone."),
            ("A", 33.4, "I am starting over tomorrow."),
            ("A", 38.0, "Subscribe if you want part two."),
        ],
        events=[(24.9, "impact")],
        moments=[Moment("line:5", "failure", 7.2, "he laments the collapse", 6.0, 3.0)],
        story={"start_line": 1, "payoff": "event:0", "end_line": 6, "title": "The box tower collapsed",
               "hook": "one more box", "emotion": "shock", "highlights": ["gone"]},
        visual_events=[{"at": "event:0", "type": "object_break", "description": "the tower of boxes on the right "
                                                                              "topples and scatters on the floor",
                        "confidence": 0.9}],
        observer=lambda t: [{"kind": "object", "description": "tower of stacked boxes", "cells":
                             ["top-right", "middle-right", "bottom-right"], "importance": "high"}],
        hooks=["One more box was too many", "The tower did not survive"],
        extra={"fall_event": 0, "walk": [16.2, 23.0]},
        expect={"visual_payoff": True},
    )


def ui_screen() -> Scenario:
    return Scenario(
        name="ui_screen", kind="ui", duration=54.0,
        speakers={"A": Speaker("en-us+m3", 160, 40, None)},
        lines=[
            ("A", 1.0, "Today I am cleaning up the old invoices."),
            ("A", 5.6, "This dashboard has every record since launch."),
            ("A", 10.6, "I will select the duplicate rows first."),
            ("A", 15.4, "Then I just press the delete button."),
            ("A", 20.0, "Wait, why is it selecting everything?"),
            ("A", 26.2, "No no no, it deleted all the data!"),
            ("A", 31.8, "I need to call the support team now."),
            ("A", 37.0, "Lesson learned, always make backups."),
        ],
        events=[(24.2, "impact")],
        moments=[Moment("line:5", "failure", 8.8, "all data deleted by accident", 10.0, 4.0, "both")],
        story={"start_line": 1, "payoff": "line:5", "end_line": 6, "title": "He deleted all the data",
               "hook": "cleaning invoices", "emotion": "shock", "highlights": ["deleted"]},
        visual_events=[{"at": "event:0", "type": "sudden_change", "description": "a red FATAL ERROR dialog appears: "
                                                                               "All data deleted", "confidence": 0.85}],
        observer=lambda t: [{"kind": "screen", "description": "dashboard table with invoice rows", "cells":
                             ["top-center", "center", "top-right", "middle-right"], "importance": "high"}],
        hooks=["One click deleted everything"],
        extra={"error_event": 0},
        expect={"layout": "screen_content", "full_fit": True},
    )


def difficult_entity() -> Scenario:
    return Scenario(
        name="difficult_entity", kind="talk", duration=56.0,
        speakers={"A": Speaker("en-us+m3", 165, 40, PersonPlacement(0, 0.29, 0.16, 0.62, phase=0.3)),
                  "B": Speaker("en-us+f2", 170, 70, PersonPlacement(1, 0.71, 0.18, 0.60, phase=1.7))},
        lines=[
            ("A", 1.0, "Welcome back, today my guest is here."),
            ("B", 4.6, "Hi everyone, happy to be here."),
            ("A", 8.2, "Tyla, you said you have fifteen cats?"),
            ("B", 12.6, "Actually it is twenty cats now."),
            ("A", 16.4, "Twenty cats in one apartment?"),
            ("B", 19.8, "And they all sleep in my bed."),
            ("A", 23.4, "Chat, Tyla said all twenty sleep in her bed!"),
            ("B", 28.4, "It is very warm, trust me."),
            ("A", 32.0, "I need a minute to process this."),
            ("B", 36.0, "Okay, next question please."),
        ],
        events=[(27.2, "cheer")],
        moments=[Moment("line:6", "ridiculous_claim", 8.2, "all twenty cats sleep in her bed", 12.0, 3.0),
                 Moment("line:3", "reveal", 6.8, "twenty cats, not fifteen", 3.0, 2.0)],
        story={"start_line": 2, "payoff": "line:6", "end_line": 8, "title": "Twenty cats, one bed",
               "hook": "the cat count", "emotion": "funny", "highlights": ["twenty"]},
        hooks=["Fifteen cats was a lie", "The cat count keeps rising"],
        asr={"transcribe_primary": {"Tyla,": "Tyler,", "Tyla": "Tyler"},
             "timing": {"Tyla,": "Tyler,", "Tyla": "Tyler"},
             "transcribe_fast": {"Tyla,": "Tyler,", "Tyla": "Tyler"},
             "transcribe_crosscheck": {"fifteen": "fifty"}},
        entities=("Tyla",),
        speaker_names=(("S2", "Tyla"),),
        expect={"name_lock": {"Tyler,": "Tyla,"}},
    )


ALL = [single_speaker, two_speakers, multi_interruption, gameplay_facecam, irl_object, ui_screen, difficult_entity]
