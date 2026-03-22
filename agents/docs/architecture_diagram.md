# Agent Pipeline — Architecture Diagram

## Main Graph Flow

```mermaid
graph TD
    START((START)) --> match_schemas

    match_schemas["🔗 match_schemas<br/><i>LLM schema alignment</i>"]
    profile_data["📊 profile_data<br/><i>Nulls, dtypes, row counts</i>"]
    normalization_node["🔄 normalization_node<br/><i>LLM-driven NormalizationSpec</i>"]
    run_blocking_tester["🧱 run_blocking_tester<br/><i>Standard/Token/Sorted/Embedding</i>"]
    run_matching_tester["🎯 run_matching_tester<br/><i>RuleBased/ML matcher</i>"]
    pipeline_adaption["⚙️ pipeline_adaption<br/><i>LLM code generation + tools</i>"]
    execute_pipeline["▶️ execute_pipeline<br/><i>Subprocess execution</i>"]
    evaluation_node["📏 evaluation_node<br/><i>LLM eval code + execution</i>"]
    investigator_node["🔍 investigator_node<br/><i>Multi-turn investigation agent</i>"]
    human_review_export["📋 human_review_export<br/><i>Source lineage tables</i>"]
    sealed_final_test["🔒 sealed_final_test_evaluation<br/><i>Held-out test set</i>"]
    save_results["💾 save_results<br/><i>Final output + report</i>"]

    match_schemas --> profile_data
    profile_data --> normalization_node
    normalization_node --> run_blocking_tester
    run_blocking_tester --> run_matching_tester
    run_matching_tester --> pipeline_adaption

    pipeline_adaption -->|"tool calls pending<br/>(max 6 loops)"| pipeline_adaption
    pipeline_adaption -->|"no more tools"| execute_pipeline

    execute_pipeline -->|"success"| evaluation_node
    execute_pipeline -->|"error + retries left"| pipeline_adaption
    execute_pipeline -->|"error + exhausted"| END_FAIL((END))

    evaluation_node -->|"success"| investigator_node
    evaluation_node -->|"error + retryable"| evaluation_node
    evaluation_node -->|"error + exhausted"| END_EVAL((END))

    investigator_node -->|"normalization_node"| normalization_node
    investigator_node -->|"run_blocking_tester"| run_blocking_tester
    investigator_node -->|"run_matching_tester"| run_matching_tester
    investigator_node -->|"pipeline_adaption"| pipeline_adaption
    investigator_node -->|"human_review_export<br/>(quality gate OR max attempts)"| human_review_export

    human_review_export -->|"val+test sets exist"| sealed_final_test
    human_review_export -->|"no test set"| save_results

    sealed_final_test --> save_results
    save_results --> END_OK((END))

    style match_schemas fill:#e1f5fe
    style profile_data fill:#e1f5fe
    style normalization_node fill:#fff3e0
    style run_blocking_tester fill:#e8f5e9
    style run_matching_tester fill:#e8f5e9
    style pipeline_adaption fill:#fce4ec
    style execute_pipeline fill:#fce4ec
    style evaluation_node fill:#f3e5f5
    style investigator_node fill:#fff9c4
    style human_review_export fill:#e0f2f1
    style sealed_final_test fill:#e0f2f1
    style save_results fill:#e0f2f1
```

## Investigator Node Detail

```mermaid
graph TD
    INV_START((investigator_node<br/>entry)) --> eval_decision

    eval_decision["1. Evaluation Decision<br/><i>Regression guard, best-metrics tracking</i>"]
    acceptance["2. Normalization Acceptance<br/><i>Did last normalization help?</i>"]
    learning["3. Learning State<br/><i>EMA gain tracking, drift detection</i>"]
    probes["4. Run 11 Probes<br/><i>Pre-computed evidence</i>"]
    early_exit{"5. Early Exit?<br/>acc≥85% OR<br/>attempts≥4"}
    cluster["6. Cluster Analysis<br/><i>Rate 5 post-clustering algorithms</i>"]
    build_guidance["7. Build Fusion Guidance<br/><i>Mismatch classifications</i>"]
    llm_loop["8. LLM Investigation Loop<br/><i>Multi-turn: code + decide</i><br/>(max 4 turns)"]
    enrich["9. Enrich Attribute Strategies<br/><i>Probe + LLM recommendations</i>"]
    safety["10. Safety Overrides<br/><i>3 hardcoded guards</i>"]
    save_transcript["11. Save & Record<br/><i>Transcript + history</i>"]

    eval_decision --> acceptance
    acceptance --> learning
    learning --> probes
    probes --> early_exit

    early_exit -->|"yes"| save_transcript
    early_exit -->|"no"| cluster

    cluster --> build_guidance
    build_guidance --> llm_loop
    llm_loop --> enrich
    enrich --> safety
    safety --> save_transcript

    save_transcript --> INV_END((return<br/>routing decision))

    style eval_decision fill:#e3f2fd
    style probes fill:#fff3e0
    style llm_loop fill:#fce4ec
    style enrich fill:#f3e5f5
    style safety fill:#ffebee

    subgraph "Probe Registry (11 probes)"
        p1["reason_distribution"]
        p2["worst_attributes"]
        p3["mismatch_sampler ★"]
        p4["attribute_improvability"]
        p5["source_attribution ★"]
        p6["null_patterns"]
        p7["correspondence_density"]
        p8["blocking_recall"]
        p9["fusion_size"]
        p10["recent_mismatches"]
        p11["directive_coverage"]
    end
```

## LLM Investigation Loop Detail

```mermaid
graph TD
    LOOP_START((LLM receives<br/>all evidence)) --> turn1

    turn1["Turn 1: LLM analyzes evidence<br/><i>metrics + probes + code + history</i>"]
    turn1 -->|'action: investigate'| exec1["Execute diagnostic Python<br/><i>subprocess, 120s timeout</i>"]
    turn1 -->|'action: decide'| decision["Final Decision<br/><i>next_node + diagnosis + recommendations</i>"]

    exec1 --> turn2["Turn 2: LLM sees code results"]
    turn2 -->|'action: investigate'| exec2["Execute more code"]
    turn2 -->|'action: decide'| decision

    exec2 --> turn3["Turn 3: LLM sees results"]
    turn3 -->|'action: investigate'| exec3["Execute code (last chance)"]
    turn3 -->|'action: decide'| decision

    exec3 --> turn4["Turn 4: LLM must decide"]
    turn4 --> decision

    decision --> targets{"Routing Target"}
    targets -->|"normalization_node"| norm_out["Fix data formats"]
    targets -->|"run_blocking_tester"| block_out["Re-test blocking"]
    targets -->|"run_matching_tester"| match_out["Re-test matching"]
    targets -->|"pipeline_adaption"| pipe_out["Fix fusion strategy"]
    targets -->|"human_review_export"| review_out["Accept + export"]

    style turn1 fill:#e3f2fd
    style decision fill:#c8e6c9
    style exec1 fill:#fff3e0
    style exec2 fill:#fff3e0
    style exec3 fill:#fff3e0
```

## Code Guardrails Pipeline

```mermaid
graph LR
    CODE_IN["LLM-generated<br/>pipeline code"] --> g1

    g1["1. include_singletons<br/>= True"]
    g2["2. Threshold<br/>freezing"]
    g3["3. list_strategy<br/>validation"]
    g4["4. Fusion guidance<br/>enforcement"]
    g5["5. Per-attribute<br/>trust_map injection"]
    g6["6. Import<br/>safety scan"]
    g7["7. EmbeddingBlocker<br/>.block()→.materialize()"]
    g8["8. Output path<br/>rewriting"]
    g9["9. eval_id column<br/>injection"]
    g10["10. ML post-clustering<br/>safety (if ml mode)"]

    g1 --> g2 --> g3 --> g4 --> g5 --> g6 --> g7 --> g8 --> g9 --> g10

    g10 --> CODE_OUT["Guardrail-safe<br/>pipeline code"]

    style CODE_IN fill:#ffcdd2
    style CODE_OUT fill:#c8e6c9
```

## Scaffold System (Cycle 2+)

```mermaid
graph TD
    cycle1["Cycle 1: Full pipeline<br/><i>LLM generates everything</i>"]
    extract["Extract scaffold<br/><i>Regex landmark detection</i>"]

    subgraph "Frozen (locked)"
        frozen1["Imports"]
        frozen2["Dataset loading"]
        frozen3["Blocking execution"]
        frozen4["Matching execution"]
        frozen5["Correspondence saving"]
    end

    subgraph "Mutable (regenerated)"
        mutable1["Post-clustering strategy"]
        mutable2["Fusion resolvers + trust_maps"]
        mutable3["add_attribute_fuser calls"]
    end

    cycle1 -->|"success"| extract
    extract --> frozen1
    extract --> mutable1

    cycle2["Cycle 2+: LLM generates<br/>ONLY mutable section"]
    splice["Splice into frozen scaffold"]

    cycle2 --> splice
    frozen1 --> splice
    mutable1 --> splice
    splice --> execute["Execute patched pipeline"]

    style frozen1 fill:#e8f5e9
    style frozen2 fill:#e8f5e9
    style frozen3 fill:#e8f5e9
    style frozen4 fill:#e8f5e9
    style frozen5 fill:#e8f5e9
    style mutable1 fill:#fff3e0
    style mutable2 fill:#fff3e0
    style mutable3 fill:#fff3e0
```

## Source Attribution Probe Flow

```mermaid
graph LR
    val["Validation Set<br/><i>25 gold records</i>"]
    fused["Fused Output<br/><i>30K+ records</i>"]
    meta["_fusion_metadata<br/><i>per-attr source inputs</i>"]

    val --> match["ID Match<br/><i>_fusion_sources → val IDs</i>"]
    fused --> match
    fused --> meta

    match --> compare["Per-Source Comparison<br/><i>exact + fuzzy match rates</i>"]
    meta --> compare

    compare --> recommend["Resolver Recommendation<br/><i>gap > 15%: prefer_higher_trust<br/>gap < 10%: type-aware default</i>"]

    recommend --> output["attribute_strategies<br/><i>per-attr resolver + trust_map</i>"]

    style val fill:#e3f2fd
    style fused fill:#fff3e0
    style output fill:#c8e6c9
```
