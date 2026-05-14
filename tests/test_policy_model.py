#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for policy model registration and manager initialization."""

from zrt.policy_model.policy_model_manager import PolicyModelManager
from zrt.policy_model.policy_register import PolicyType, register_model


def test_policy_model_registration_and_initialization():
    register_model()
    manager = PolicyModelManager()

    policy_types = {
        PolicyType.PRIORITY,
        PolicyType.OOTB_PERFORMANCE,
        PolicyType.OPERATOR_OPTIMIZATION,
        PolicyType.SYSTEM_DESIGN,
    }

    assert policy_types.issubset(manager.target_model_map)
    assert policy_types.issubset(manager.policy_models_map)


if __name__ == "__main__":
    test_policy_model_registration_and_initialization()
