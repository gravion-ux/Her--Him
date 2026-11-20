"""Tests for the layered memory store and the recall scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from herhim.memory.recall import Recall, recall
from herhim.memory.store import (
    CONFIDENCE_LEVELS,
    RECENT_LIMIT,
    MemoryRecord,
    MemoryStore,
)


class TestMemoryStoreWriting:
    """Tests for writing into the memory store."""

    def test_unknown_confidence_is_refused(self) -> None:
        """A confidence level outside the vocabulary is refused."""
        with pytest.raises(ValueError, match="confidence"):
            MemoryStore().add("facts", "text", confidence="maybe")

    def test_confidence_levels(self) -> None:
        """The confidence vocabulary is the documented one."""
        assert CONFIDENCE_LEVELS == ("verified", "assumed", "open_question")

    def test_add_fact(self) -> None:
        """A verified fact lands in the facts layer."""
        store = MemoryStore()
        record = store.add_fact("The user lives in Berlin", source="profile")
        assert record.layer == "facts"
        assert record.confidence == "verified"
        assert store.facts()[0].text == "The user lives in Berlin"

    def test_add_profile(self) -> None:
        """A profile note lands in the profile layer."""
        store = MemoryStore()
        store.add_profile("Prefers short replies")
        assert store.profile()[0].text == "Prefers short replies"

    def test_add_episode(self) -> None:
        """An episode summary lands in the episodes layer."""
        store = MemoryStore()
        store.add_episode("We talked about work")
        assert store.episodes()[0].text == "We talked about work"

    def test_add_turn(self) -> None:
        """A verbatim turn lands in the recent layer with its role."""
        store = MemoryStore()
        record = store.add_turn("user", "hello", source="user")
        assert record.layer == "recent"
        assert record.role == "user"

    def test_assumptions_are_not_facts(self) -> None:
        """An assumption is not returned as a verified fact."""
        store = MemoryStore()
        store.add("facts", "maybe", confidence="assumed")
        assert store.facts() == []

    def test_recent_layer_is_bounded(self) -> None:
        """The recent layer keeps only the most recent turns."""
        store = MemoryStore()
        for index in range(RECENT_LIMIT + 5):
            store.add_turn("user", f"turn {index}")
        assert len(store.recent()) == RECENT_LIMIT
        assert store.recent()[-1].text == f"turn {RECENT_LIMIT + 4}"

    def test_clear(self) -> None:
        """Clearing forgets everything."""
        store = MemoryStore()
        store.add_fact("something")
        store.clear()
        assert store.records == []


class TestMemoryStoreReading:
    """Tests for reading from the memory store."""

    def test_layer_selection(self) -> None:
        """Each layer returns only its own records."""
        store = MemoryStore()
        store.add_fact("a fact")
        store.add_profile("a profile")
        store.add_episode("an episode")
        assert len(store.layer("facts")) == 1
        assert len(store.layer("profile")) == 1
        assert len(store.layer("episodes")) == 1

    def test_recent_respects_the_limit(self) -> None:
        """The recent view honours its limit argument."""
        store = MemoryStore()
        for index in range(5):
            store.add_turn("user", f"turn {index}")
        assert len(store.recent(limit=2)) == 2

    def test_recent_is_oldest_first(self) -> None:
        """The recent view keeps the original ordering."""
        store = MemoryStore()
        store.add_turn("user", "first")
        store.add_turn("user", "second")
        assert [record.text for record in store.recent()] == ["first", "second"]


class TestMemoryPersistence:
    """Tests for saving and loading the memory store."""

    def test_to_dict(self) -> None:
        """The serialised store carries every record."""
        store = MemoryStore()
        store.add_fact("a fact")
        assert len(store.to_dict()["records"]) == 1

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """A saved store loads back with its records."""
        path = tmp_path / "memory.json"
        store = MemoryStore(path)
        store.add_fact("a fact", source="profile")
        store.save()
        loaded = MemoryStore.load(path)
        assert loaded.facts()[0].text == "a fact"
        assert loaded.facts()[0].source == "profile"

    def test_save_to_an_explicit_path(self, tmp_path: Path) -> None:
        """Saving accepts a path of its own."""
        store = MemoryStore()
        store.add_fact("a fact")
        target = store.save(tmp_path / "other.json")
        assert target is not None
        assert target.exists()

    def test_save_without_a_path_returns_none(self) -> None:
        """A store with no path cannot be saved."""
        assert MemoryStore().save() is None

    def test_load_of_a_missing_file_is_empty(self, tmp_path: Path) -> None:
        """Loading a store that does not exist yields no records."""
        assert MemoryStore.load(tmp_path / "nope.json").records == []

    def test_load_of_a_broken_file_is_empty(self, tmp_path: Path) -> None:
        """Loading a store that is not valid JSON yields no records."""
        path = tmp_path / "memory.json"
        path.write_text("{not json", encoding="utf-8")
        assert MemoryStore.load(path).records == []

    def test_record_from_dict(self) -> None:
        """A record round-trips through a mapping."""
        record = MemoryRecord("r-1", "facts", "text", "src", "verified", 1.5, "user")
        restored = MemoryRecord.from_dict(record.to_dict())
        assert restored == record

    def test_record_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys in a stored record are ignored."""
        payload = {"record_id": "r-1", "text": "text", "nonsense": 1}
        assert MemoryRecord.from_dict(payload).layer == "facts"

    def test_saved_store_is_valid_json(self, tmp_path: Path) -> None:
        """The saved store is a JSON document."""
        path = tmp_path / "memory.json"
        store = MemoryStore(path)
        store.add_fact("a fact")
        store.save()
        assert json.loads(path.read_text(encoding="utf-8"))["records"]


class TestRecall:
    """Tests for choosing what to put in front of the model."""

    def test_empty_query_recalls_nothing(self) -> None:
        """A query with no words recalls nothing."""
        store = MemoryStore()
        store.add_fact("The user lives in Berlin")
        assert recall(store, "   ") == []

    def test_no_candidates_recalls_nothing(self) -> None:
        """An empty store recalls nothing."""
        assert recall(MemoryStore(), "Berlin") == []

    def test_matching_fact_is_recalled(self) -> None:
        """A fact that shares words with the query is recalled."""
        store = MemoryStore()
        store.add_fact("The user lives in Berlin")
        store.add_fact("The user dislikes rain")
        recalled = recall(store, "Where does the user live? Berlin")
        assert recalled
        assert "Berlin" in recalled[0].record.text

    def test_unrelated_records_are_skipped(self) -> None:
        """A record with no shared words is not recalled."""
        store = MemoryStore()
        store.add_fact("The user lives in Berlin")
        assert recall(store, "quantum chromodynamics") == []

    def test_source_is_preserved(self) -> None:
        """A recalled item keeps the source it came from."""
        store = MemoryStore()
        store.add_fact("The user lives in Berlin", source="profile")
        assert recall(store, "Berlin")[0].record.source == "profile"

    def test_limit_is_respected(self) -> None:
        """The recall honours its limit argument."""
        store = MemoryStore()
        for index in range(5):
            store.add_fact(f"Berlin fact {index}")
        assert len(recall(store, "Berlin", limit=2)) == 2

    def test_verbatim_turns_are_not_recalled(self) -> None:
        """The recent layer is the session's business, not the recall's."""
        store = MemoryStore()
        store.add_turn("user", "Berlin is nice")
        assert recall(store, "Berlin") == []

    def test_recall_serialises(self) -> None:
        """A recalled item serialises its record and its score."""
        store = MemoryStore()
        store.add_fact("The user lives in Berlin", source="profile")
        item = Recall(record=store.facts()[0], score=0.5)
        data = item.to_dict()
        assert data["source"] == "profile"
        assert data["score"] == 0.5
