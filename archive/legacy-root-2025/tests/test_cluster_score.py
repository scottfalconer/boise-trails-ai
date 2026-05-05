from trail_route_ai.challenge_planner import ClusterScore


def test_isolation_and_completion_affect_score():
    base = ClusterScore(10, 20, 0.0, 0.0, 0.0)
    isolated = ClusterScore(10, 20, 5.0, 0.0, 0.0)
    completed = ClusterScore(10, 20, 0.0, 1.0, 0.0)
    assert isolated.total_score < base.total_score
    assert completed.total_score < base.total_score


def test_effort_distribution_penalty():
    balanced = ClusterScore(10, 20, 0.0, 0.0, 0.0)
    unbalanced = ClusterScore(10, 20, 0.0, 0.0, 5.0)
    assert balanced.total_score < unbalanced.total_score
