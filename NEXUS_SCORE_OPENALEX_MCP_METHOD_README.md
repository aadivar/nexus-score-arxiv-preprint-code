# Nexus Score OpenAlex-MCP Methodology

## Measuring AI-readiness of scholarly records through controlled metadata withholding, restoration, grounding, misattribution, cost, and visibility experiments

### One-line thesis

AI cannot ground what the scholarly record cannot identify. Nexus Score measures whether a scholarly record exposes enough identifiers, provenance links, access paths, and entity relations for AI systems to retrieve, attribute, verify, refuse, and reuse scientific claims responsibly.

### Core claim

This study does not compare OpenAlex against Crossref, Scopus, Web of Science, Semantic Scholar, or any other scholarly index. It uses OpenAlex as a single open scholarly knowledge graph and experimentally varies how much of the OpenAlex metadata substrate is exposed to an AI agent through an MCP interface. The purpose is to test whether metadata depth itself controls AI grounding, attribution, refusal, misattribution, hallucination, cost, and visibility.

The experiment is designed around a simple causal setup:

```text
same scholarly work
same question
same model
same MCP interface
same tool budget
same output schema
different metadata view
```

If the same work produces different outcomes when specific metadata facets are hidden or restored, the experiment is no longer about one provider being better than another. It is about whether metadata fields and entity edges are necessary for accountable AI-mediated research.

The primary expected result is not merely that richer metadata improves retrieval. The stronger claim is that missing metadata produces a specific accountability failure: the AI system may cite real papers and real entities while attaching them through false author, institution, funder, citation, or access edges. This is misattribution. In scholarly AI, misattribution is often more dangerous than obvious hallucination because the paper exists, the identifier may resolve, and the answer still gives a false account of who did the work, where it was done, who funded it, or what source supports the claim.

The sustainability claim is also causal: if a missing identifier is repaired in the open scholarly substrate, that repair can be reused by every downstream tool. If the identifier is not repaired, every AI system must attempt recovery again through model memory, web search, or additional compute. Substrate repair amortizes; agent recovery repeats.

The equity claim is that metadata poverty can become AI invisibility. Records with richer identifiers, affiliations, funder links, citation edges, and access routes are easier for AI systems to retrieve, verify, cite, and reuse. Records with weaker metadata can drop out of AI-mediated discovery even when they are topically relevant. This is the Matthew effect applied to the scholarly metadata substrate: already well-described research becomes more visible, while poorly described research becomes less visible.

---

## Research questions

The study is organized around six research questions.

First, does Nexus Score predict whether an AI agent can produce a grounded scholarly answer?

Second, do the five Nexus facets predict the corresponding task outcomes? In other words, does People metadata help author attribution, Organizations metadata help institution attribution, Funding metadata help funder attribution, Provenance metadata help source and citation tracing, and Access metadata help source-inspectable answers?

Third, when relevant metadata is withheld, does the agent fail by refusing, hallucinating, returning nothing, selecting the wrong real entity, or misattributing real entities through false scholarly edges?

Fourth, can LLM training-data priors, open web search, or higher compute substitute for missing authoritative metadata?

Fifth, does metadata restoration repair failures more reliably and cheaply than agentic recovery?

Sixth, does AI-mediated discovery amplify a Matthew effect in which high-Nexus records become overrepresented in retrieved, cited, and synthesized outputs while low-Nexus records become underrepresented?

---

## Non-goals

This study does not attempt to prove that OpenAlex is perfect. It uses OpenAlex because it is a large open scholarly knowledge graph with works, authors, institutions, funders, topics, citations, identifiers, and access metadata that can be queried reproducibly.

This study does not compare OpenAlex with Crossref. Cross-provider comparison creates an avoidable interpretive problem: reviewers can dismiss the result as a source comparison rather than a metadata-depth experiment.

This study does not use embeddings. The goal is not to optimize semantic search. The goal is to isolate whether structured metadata and graph edges enable accountable grounding. Retrieval is performed through search, filters, identifiers, citation links, topic fields, and MCP-accessible OpenAlex tools.

This study does not claim that metadata alone solves scientific truth. Metadata does not prove that a paper is correct. Metadata makes claims attributable, inspectable, auditable, and refusably grounded.

This study does not claim that LLMs can never help. It tests a narrower and more empirical claim: within the tested regimes, LLM priors, web search, and higher compute are less reliable and less reusable than authoritative metadata restoration for identifier-dependent scholarly grounding.

---

## Study substrate

The study uses OpenAlex as the single scholarly substrate.

The preferred implementation is a local OpenAlex snapshot exposed through a custom MCP server. A live OpenAlex API-backed MCP server can also be used for development and demonstration, but the main experimental run should use a frozen snapshot so the results can be reproduced.

The OpenAlex snapshot is divided into two logical layers.

The first layer is the hidden truth store. This contains the full OpenAlex record for each sampled work, including work IDs, DOIs, author IDs, ORCIDs where present, institution IDs, RORs where present, funder IDs, grants where present, referenced works, topics, source metadata, publication year, open access metadata, and locations. The AI agent never sees the hidden truth store directly. It is used only for dataset construction, metadata view generation, and mechanical adjudication.

The second layer is the experimental MCP view. This is the version of the record exposed to the AI agent during a run. Different views expose different amounts of metadata. The same underlying scholarly work can appear as a full record, a minimal record, a People-masked record, an Organizations-masked record, a Funding-masked record, a Citation-masked record, an Access-masked record, or a minimal record with exactly one facet restored.

This separation between hidden truth and exposed view is essential. It allows the study to know the correct answer while controlling exactly what the AI agent is allowed to use.

---

## MCP interface

The MCP server is the only structured data interface used by the AI agent in the OpenAlex-only arms. It should expose scholarly operations rather than raw database access.

Recommended tools include:

```text
search_works
get_work
search_authors
get_author
search_institutions
get_institution
search_funders
get_funder
search_sources
get_source
search_topics
get_topic
get_works_by_author
get_works_by_institution
get_works_by_funder
get_referenced_works
get_citing_works
get_topic_works
get_open_access_locations
verify_work_entity_edge
verify_identifier
```

The MCP layer must support capability masking. If a metadata facet is hidden in the assigned view, tools and filters that depend on that facet must also be disabled. This is critical.

For example, in the Organizations-masked view, the agent should not receive ROR IDs, OpenAlex institution IDs, structured institution edges, country fields, or institution-based filters. If the fields are hidden but the agent can still call `get_works_by_institution`, then the metadata has not truly been withheld. The experiment would be invalid.

Similarly, in the People-masked view, the agent may see raw author names if the view permits it, but it should not be able to filter by ORCID or OpenAlex author ID. In the Funding-masked view, it should not be able to call funder-specific filters. In the Citation-masked view, it should not be able to traverse references or citing works. In the Access-masked view, it should not receive open access status, license, repository links, or full-text locations.

The MCP server should log every tool call, every returned object, every filtered-out field, and every unavailable capability. These logs become part of the reproducibility record.

---

## Nexus Score

Nexus Score is a metadata-substrate readiness measure. It estimates whether a scholarly record exposes enough persistent identifiers, provenance links, access paths, and entity relations for AI systems to retrieve, attribute, verify, refuse, and reuse scientific claims responsibly.

Nexus Score is not an impact score. It is not a journal ranking. It is not a publisher grade. It is not a scientific merit score. It measures the record surrounding the work, not the value of the work itself.

The score has five facets.

### Provenance

Provenance measures whether the work can be traced as a stable scholarly object. Signals include OpenAlex work ID, DOI, source, publication date, publication type, references, citation links, and version or source metadata.

### People

People measures whether contributors can be disambiguated. Signals include author OpenAlex IDs, ORCIDs where available, structured authorship edges, author order, contribution links, and the ability to retrieve works by a resolved author entity rather than by a name string.

### Organizations

Organizations measures whether affiliations and institutions can be resolved. Signals include institution OpenAlex IDs, ROR IDs where available, structured affiliation edges, institution country, and the ability to retrieve works by a resolved institution rather than by raw affiliation text.

### Funding

Funding measures whether financial support and responsibility can be traced. Signals include funder OpenAlex IDs, funder identifiers, grant or award IDs where available, and work-to-funder or work-to-award edges.

### Access

Access measures whether the work can be inspected or reused. Signals include open access status, license, repository links, best open access location, locations, landing-page URLs, PDF URLs where available, and machine-readable access paths.

A record-level facet score is computed as:

```text
Nexus_facet(record) = weighted mean of:
  presence
  resolvability
  edge_specificity
  queryability
  provenance_of_metadata
```

Where:

```text
presence = the field exists
resolvability = the field resolves to a stable identifier or object
edge_specificity = the field is attached to the correct work/entity relation
queryability = the field can be used through the MCP interface
provenance_of_metadata = the source or status of the field can be inspected
```

The composite score is:

```text
Nexus(record) =
  w_provenance    × Nexus_provenance(record)
+ w_people        × Nexus_people(record)
+ w_organizations × Nexus_organizations(record)
+ w_funding       × Nexus_funding(record)
+ w_access        × Nexus_access(record)
```

The composite score is useful for ranking overall AI-readiness, but the validation should be facet-specific. The central validation claim is not simply that high Nexus Score predicts better answers. The stronger claim is that the right Nexus facet predicts the right task outcome.

The expected diagonal is:

```text
People score        → author attribution
Organizations score → institution attribution
Funding score       → funding attribution
Provenance score    → citation and source tracing
Access score        → inspectable-answer generation
```

If the diagonal appears and negative controls show little effect, Nexus Score is validated as an AI-readiness measure rather than a generic metadata-richness score.

---

## Corpus construction

The study should be built from scratch rather than bound to the earlier p53-only corpus.

The recommended design is a multi-field OpenAlex corpus stratified by topic, year, metadata depth, country or region where available, source type, open access status, and citation count.

The corpus should include multiple domains so the result is not dismissed as a biomedical artifact. Suggested strata include biomedical science, computer science, climate and environmental science, physics or materials science, economics or policy, social science, and humanities.

The corpus is divided into four pools.

### Gold-edge pool

The Gold-edge pool contains works with sufficiently rich OpenAlex metadata to serve as hidden ground truth before masking. These records should contain, where available, DOI, work ID, author ID, ORCID, institution ID, ROR, funder ID, grant or award metadata, referenced works, topic metadata, and access metadata.

This pool enables causal withholding and restoration because the correct metadata exists in the hidden truth store even when it is hidden from the agent.

### Natural low-Nexus pool

The Natural low-Nexus pool contains works that are naturally metadata-poor in OpenAlex. These records are not artificially masked. They are used to study observational AI-readiness, auditability, and Matthew-effect visibility in realistic conditions.

### Ambiguity pool

The Ambiguity pool contains works where name-string retrieval is likely to fail. Examples include common author names, similar institution names, multiple authors with the same surname, authors with multiple affiliations, institutions with branch campuses, and works in fields with sparse identifier adoption.

This pool is used to test whether missing identifiers cause misattribution.

### Equity pool

The Equity pool contains records stratified by country, region, institution type where available, source type, language where available, open access status, topic, field, publication year, and citation count. This pool is used to test whether metadata-poor records are disadvantaged in AI-mediated discovery and synthesis.

The final sample should be balanced enough that high-Nexus and low-Nexus records exist within the same topics, years, and citation bands. This is necessary to separate metadata visibility from topical popularity and citation prestige.

---

## Metadata views

The central experiment creates multiple views of the same OpenAlex records. Each view is a controlled projection of the hidden truth store.

### V_full

`V_full` exposes all available OpenAlex metadata for the work.

It includes title, abstract where available, DOI, OpenAlex work ID, authors, author IDs, ORCID where available, institutions, institution IDs, ROR where available, funders, funder IDs, grants where available, references, topics, source metadata, open access metadata, and access locations.

### V_minimal

`V_minimal` exposes only a minimal bibliographic record.

It includes title, year, source, and abstract where available. DOI may be included or withheld depending on whether the task is a known-item control or an attribution task. It does not expose author IDs, ORCIDs, institution IDs, RORs, funder IDs, grant IDs, citation graph edges, topics beyond a minimal label, or access links.

### V_people_masked

`V_people_masked` exposes the record while hiding person-resolution metadata.

It hides author OpenAlex IDs, ORCIDs, structured author-work edges, and author-based filters. It may retain raw author names if the experiment is testing name-string ambiguity.

### V_organizations_masked

`V_organizations_masked` hides institution-resolution metadata.

It hides institution OpenAlex IDs, RORs, structured affiliation edges, institution-country fields, and institution-based filters. It may retain raw affiliation strings if the experiment is testing whether raw strings are enough.

### V_funding_masked

`V_funding_masked` hides funder and grant metadata.

It hides funder IDs, award IDs, grant IDs, funder-work edges, and funder-based filters.

### V_citation_masked

`V_citation_masked` hides citation and reference metadata.

It hides referenced works, citing works, citation graph traversal, and reference-list filters.

### V_access_masked

`V_access_masked` hides access metadata.

It hides open access status, licenses, repository links, best open access locations, landing-page URLs, PDF URLs, and full-text locations.

### V_restored_facet

`V_restored_facet` starts from `V_minimal` and restores exactly one metadata facet.

The restored views are:

```text
V_minimal_plus_provenance
V_minimal_plus_people
V_minimal_plus_organizations
V_minimal_plus_funding
V_minimal_plus_access
```

These are the most important experimental views because they isolate repair. The restoration experiment asks whether adding back the relevant facet repairs the specific failure that appeared when the facet was absent.

---

## Task classes

Each task class is designed to depend on one or more Nexus facets. The task set should include both metadata-dependent tasks and negative controls.

### Known-item control

The known-item task asks the agent to find a specific paper from an exact DOI, exact title, or title plus year.

Expected metadata dependency is low. This task functions as a negative control. If Nexus Score strongly changes known-item lookup, the score may be acting as a generic visibility or prominence proxy rather than an AI-readiness measure.

### Author attribution

The author-attribution task asks the agent to find or verify a paper by a specific author on a topic or claim.

Expected dependency is the People facet. Ground truth is the author-work edge, preferably involving a resolved author ID and ORCID where available.

### Institution attribution

The institution-attribution task asks the agent to find or verify work from a specific institution on a topic or claim.

Expected dependency is the Organizations facet. Ground truth is the work-institution edge, preferably involving institution ID and ROR where available.

### Funding attribution

The funding-attribution task asks the agent to find or verify work funded by a specific funder, award, or grant.

Expected dependency is the Funding facet. Ground truth is the work-funder or work-award edge.

### Citation lineage

The citation-lineage task asks the agent to find works citing a target work, works cited by a target work, or papers that build on a specific prior work.

Expected dependency is Provenance plus citation graph structure. Ground truth is the referenced-work or citing-work edge.

### Access-grounded answer

The access-grounded task asks the agent to answer using a source that can be inspected.

Expected dependency is the Access facet. Ground truth is the presence of an open access location, source URL, license, repository record, or other inspectable access path.

### Ambiguous entity disambiguation

The ambiguous-entity task asks the agent to find work by an author with a common name, an institution with ambiguous naming, or a topic involving multiple similar entities.

Expected dependency is People plus Organizations. This task is designed to expose misattribution when identifiers are hidden.

### Literature review visibility

The literature-review task asks the agent to return a set of important, recent, relevant, or representative papers on a topic.

Expected dependency is overall Nexus Score, because works with richer identifiers, citation links, topics, and access paths are easier to retrieve, verify, and include. This task is used for the Matthew-effect analysis.

---

## Agent arms

The study compares several AI-access regimes. These arms test whether metadata can be replaced by model memory, web search, or more compute.

### Arm A: Closed-book LLM

The model receives no tools. It answers from training data and prompt context only.

This arm tests whether model memory can substitute for metadata.

### Arm B: OpenAlex MCP RAG

The model may use only the OpenAlex MCP server. It must ground every claim in MCP-returned evidence.

This is the primary experimental arm.

### Arm C: OpenAlex MCP RAG plus LLM prior

The model may use its internal knowledge to propose candidates, but every final claim must be verified through the OpenAlex MCP server.

This arm tests whether training-data priors help when verification is still required.

### Arm D: Open web search only

The model may use open web search but not the OpenAlex MCP server.

This arm tests the identifier-recovery path outside the scholarly substrate.

### Arm E: OpenAlex MCP plus open web search

The model may use both the OpenAlex MCP server and web search.

This arm tests whether web search can repair missing metadata in the assigned view.

### Arm F: High-compute recovery

The model receives the same available metadata as the weaker view but is given more tool calls, more search calls, more retries, and a larger token budget.

This arm tests whether additional compute closes the metadata gap.

### Arm G: Guarded MCP

The model may answer only when the relevant edge is verified by the MCP server. If the edge is not available in the assigned view, it must refuse.

This arm tests whether verification policy can convert misattribution into clean refusal.

---

## Prompt and output protocol

Every run uses the same prompt family, same tool budget, same model family, same temperature, and same output schema within a given arm.

The model must return structured JSON rather than unconstrained prose.

Recommended output schema:

```json
{
  "answer_status": "ANSWERED | REFUSED | NO_RESULT",
  "task_class": "known_item | author_attribution | institution_attribution | funding_attribution | citation_lineage | access_grounding | ambiguous_entity | literature_review",
  "work_id": "",
  "doi": "",
  "author_id": "",
  "orcid": "",
  "institution_id": "",
  "ror": "",
  "funder_id": "",
  "evidence_work_ids": [],
  "evidence_edges": [],
  "evidence_tool_calls": [],
  "confidence": 0.0,
  "reason_for_refusal": ""
}
```

The model is instructed that a real DOI is not sufficient evidence. The relationship must also be true. For example, a real paper and a real institution do not make a correct answer unless the paper is actually linked to that institution in the assigned evidence view or can be verified through the permitted tools.

---

## Mechanical adjudication

Adjudication uses the hidden truth store. The judge is not an LLM wherever a mechanical check is possible.

For each output, the evaluator checks:

```text
Does the returned work resolve?
Does the returned author resolve?
Does the returned institution resolve?
Does the returned funder resolve?
Does the returned citation edge exist?
Does the returned work-entity edge exist?
Was the relevant evidence available in the assigned view?
Did the model refuse when it should have refused?
Did the model answer when the required edge was absent?
```

The adjudicator records both entity validity and edge validity. This distinction is central.

A real entity can still be wrong if the edge is false. A real paper can still be misattributed if the author, institution, funder, or citation relation does not hold.

---

## Outcome taxonomy

Every answer is assigned to exactly one primary outcome bucket.

### CORRECT

The returned entity and the required relation match the hidden truth store.

### WRONG_REAL

The returned entity exists, but it is not the target answer.

### MISATTRIBUTED

The returned entities exist, but the asserted scholarly edge is false.

Examples:

```text
real paper + wrong author edge
real paper + wrong institution edge
real paper + wrong funder edge
real paper + unsupported citation edge
real author + wrong paper edge
real institution + wrong paper edge
```

This is the main failure mode the study is designed to measure.

### HALLUCINATED

The returned DOI, identifier, entity, or source does not resolve.

### UNSUPPORTED

The response may be plausible, but it does not provide a valid evidence edge through the permitted tools.

### REFUSED_CORRECTLY

The model refuses because the assigned metadata view does not contain enough evidence to answer.

### REFUSED_INCORRECTLY

The model refuses even though the assigned metadata view did contain enough evidence.

### NO_RESULT

The model produces no usable answer.

---

## Metrics

The study reports grounding, attribution, visibility, and cost metrics.

Grounding metrics:

```text
correct_edge_rate
misattribution_rate
hallucination_rate
unsupported_answer_rate
correct_refusal_rate
incorrect_refusal_rate
no_result_rate
```

Retrieval metrics:

```text
hit@1
hit@5
hit@10
MRR
candidate_entity_recall
candidate_edge_recall
```

Cost metrics:

```text
tool_calls_per_answer
web_searches_per_answer
tokens_per_answer
latency_per_answer
dollar_cost_per_answer
dollar_cost_per_correct_answer
dollar_cost_per_misattribution_avoided
```

Visibility metrics:

```text
selection_rate
exposure@10
exposure@20
Nexus_decile_share_in_corpus
Nexus_decile_share_in_outputs
visibility_lift_by_decile
Matthew_Amplification_Index
low_Nexus_dropout_rate
high_Nexus_compounding_rate
```

---

## Primary experiment 1: Metadata withholding

The metadata-withholding experiment compares full metadata to facet-masked metadata for the same works and same questions.

For each work and task:

```text
run V_full
run V_people_masked
run V_organizations_masked
run V_funding_masked
run V_citation_masked
run V_access_masked
```

The expected pattern is:

```text
People masked        → author-attribution failures rise
Organizations masked → institution-attribution failures rise
Funding masked       → funding-attribution failures rise
Citation masked      → citation-lineage failures rise
Access masked        → access-grounded answer failures rise
```

The key metric is the change in correct edge rate and misattribution rate when the relevant facet is removed.

This experiment answers the question: what fails when the accountability edge is missing?

---

## Primary experiment 2: Metadata restoration

The metadata-restoration experiment starts from a weak view and restores exactly one facet.

For each work and task:

```text
run V_minimal
run V_minimal_plus_provenance
run V_minimal_plus_people
run V_minimal_plus_organizations
run V_minimal_plus_funding
run V_minimal_plus_access
run V_full
```

The expected pattern is facet-specific repair.

Examples:

```text
Restoring People should repair author attribution more than funding attribution.
Restoring Organizations should repair institution attribution more than known-item lookup.
Restoring Funding should repair funder tracing more than author disambiguation.
Restoring Provenance should repair citation lineage more than institution attribution.
Restoring Access should repair inspectable-answer tasks more than title lookup.
```

This experiment is the strongest causal validation of Nexus Score.

It asks: does the relevant metadata repair the relevant AI failure?

---

## Primary experiment 3: Facet diagonal

The facet-diagonal analysis creates a matrix:

```text
rows    = task classes
columns = restored Nexus facets
cells   = improvement in correct grounding
```

Expected structure:

```text
                         Restored facet
Task class        Prov   People   Org   Funding   Access
Known item         low     low     low     low       low
Author             low    high     med     low       low
Institution        low     med    high     low       low
Funding            low     low     low    high       low
Citation          high     low     low     low       low
Access             med     low     low     low      high
```

The diagonal pattern is what validates Nexus Score as an AI-readiness measure.

If every restored facet improves every task equally, Nexus Score is just a generic metadata-richness index. If the correct facet improves the correct task, Nexus Score measures structured readiness.

---

## Primary experiment 4: Negative controls

Negative controls are tasks where metadata depth should have little effect.

Recommended negative controls:

```text
exact DOI lookup
exact title lookup
publication year lookup
known-item retrieval from exact title and year
```

Expected result:

```text
Nexus Score should not produce large gains on metadata-independent tasks.
```

This protects the study from the criticism that Nexus Score is merely a proxy for citation count, source prestige, paper prominence, or general discoverability.

---

## Primary experiment 5: LLM and compute substitution

This experiment tests whether model memory, web search, or higher compute can substitute for missing metadata.

For the same masked task, compare:

```text
V_masked + OpenAlex MCP RAG
V_masked + OpenAlex MCP RAG + LLM prior
V_masked + web search only
V_masked + OpenAlex MCP + web search
V_masked + high-compute recovery
V_restored + OpenAlex MCP RAG
```

The central comparison is:

```text
metadata restored + cheap MCP verification
versus
metadata missing + expensive agentic recovery
```

Report:

```text
correct_edge_rate
misattribution_rate
hallucination_rate
refusal_rate
tool_calls
web_search_calls
tokens
latency
dollar_cost_per_correct_answer
```

The claim should be phrased carefully:

```text
Within the tested regimes, additional LLM prior knowledge, web search, and higher compute do not reliably substitute for authoritative metadata edges. They may increase attempts, but they also increase cost and verification burden, and they can still produce misattribution or hallucinated identifiers.
```

The desired result is that restored metadata is cheaper, more reliable, and more accountable than recovery attempts.

---

## Matthew-effect experiment

The Matthew-effect experiment tests whether AI-mediated discovery overrepresents high-Nexus records and underrepresents low-Nexus records.

The hypothesis is:

```text
Metadata-rich records become more visible in AI-mediated discovery because they are easier to retrieve, verify, cite, and synthesize. Metadata-poor records become less visible, even when they are topically relevant.
```

This is not just a fairness claim. It is an infrastructure claim. If an AI system has to choose among many relevant papers, the ones with stronger metadata edges are safer to include. Over repeated retrieval and synthesis, that safety advantage becomes visibility advantage.

### Topic-level candidate sets

For each topic, build a relevant candidate set from OpenAlex using topic, field, subfield, year, and keyword constraints.

Each candidate set should include works across Nexus Score deciles. The set should also include controls for citation count, year, source type, open access status, and country or region where available.

### Discovery task

Ask the agent:

```text
Return 10 important recent papers on this topic.
Return a balanced set suitable for a literature review.
Include evidence for each paper.
```

Run the task under:

```text
V_full
V_minimal
facet-masked views
OpenAlex MCP RAG
OpenAlex MCP RAG + LLM prior
OpenAlex MCP + web search
high-compute recovery
```

### Visibility lift

For each topic and each Nexus Score decile, compute:

```text
available_share(decile) =
  share of relevant candidate corpus in the decile

output_share(decile) =
  share of returned or cited papers in the decile

visibility_lift(decile) =
  output_share(decile) / available_share(decile)
```

Expected result:

```text
low Nexus deciles  → visibility_lift < 1
high Nexus deciles → visibility_lift > 1
```

### Matthew Amplification Index

Define:

```text
Matthew Amplification Index =
  visibility_lift(top Nexus decile)
- visibility_lift(bottom Nexus decile)
```

A positive value means metadata-rich records are overrepresented relative to their availability while metadata-poor records are underrepresented.

### Iterated discovery

The strongest Matthew-effect design is iterative.

Round 0:

```text
Construct the full relevant candidate set.
```

Round 1:

```text
Ask the agent for 10 relevant papers.
```

Round 2:

```text
Using the selected papers, ask the agent for related or follow-up papers.
```

Round 3:

```text
Ask the agent to write a mini literature map with citations.
```

Track after each round:

```text
Nexus Score distribution
low-Nexus survival rate
high-Nexus amplification rate
country diversity
institution diversity
source diversity
open access share
misattribution rate
unsupported citation rate
```

Expected pattern:

```text
The output distribution shifts toward higher-Nexus records across rounds.
Low-Nexus records disappear earlier.
High-Nexus records compound in visibility.
```

This is the empirical version of the richer-get-richer claim.

---

## Statistical analysis

The study uses paired and hierarchical models because the same works appear under multiple metadata views and agent arms.

### Correct grounding model

```text
logit(P(CORRECT)) =
  β0
+ β1 relevant_Nexus_facet
+ β2 metadata_view
+ β3 agent_arm
+ β4 task_class
+ β5 cited_by_count
+ β6 open_access_status
+ β7 publication_year
+ β8 topic
+ random_effect(work)
+ random_effect(query_template)
+ random_effect(topic_or_field)
```

### Misattribution model

```text
logit(P(MISATTRIBUTED)) =
  β0
+ β1 missing_relevant_facet
+ β2 metadata_view
+ β3 agent_arm
+ β4 ambiguity_pool_indicator
+ β5 cited_by_count
+ β6 open_access_status
+ β7 publication_year
+ random_effect(work)
+ random_effect(query_template)
```

### Refusal model

```text
logit(P(REFUSED_CORRECTLY)) =
  β0
+ β1 evidence_available_in_view
+ β2 guarded_policy
+ β3 metadata_view
+ β4 task_class
+ random_effect(work)
+ random_effect(query_template)
```

### Visibility model

```text
logit(P(selected_for_answer)) =
  β0
+ β1 Nexus Score
+ β2 cited_by_count
+ β3 open_access_status
+ β4 publication_year
+ β5 topic_fixed_effects
+ β6 source_fixed_effects
+ β7 country_or_region
+ random_effect(query)
```

The key Matthew-effect test is whether Nexus Score predicts selection after controlling for citation count, topic, year, source, and open access status.

Report paired bootstrap confidence intervals for all primary contrasts.

Primary reported effects:

```text
metadata withholding effect
metadata restoration effect
facet-diagonal effect
misattribution reduction
correct refusal improvement
cost per correct grounded answer
cost per misattribution avoided
Matthew Amplification Index
visibility lift by Nexus decile
```

---

## Reproducibility protocol

The study should be preregistered before the main run.

Freeze:

```text
OpenAlex snapshot date
corpus construction queries
sampling seed
topic strata
Nexus Score formula
metadata view definitions
MCP tool permissions
question templates
agent arms
model versions
tool budgets
output schema
adjudication rules
primary endpoints
statistical models
Matthew-effect metrics
```

Log:

```text
every MCP call
every web search call
every returned object
every masked field
every refused tool capability
every model response
every adjudication decision
every cost estimate
```

Release:

```text
code
configuration
view-generation scripts
question templates
MCP server specification
adjudication scripts
derived result tables
figures
README
limitations
```

If raw records cannot be redistributed in full, release record IDs, snapshot date, extraction scripts, and derived metrics.

---

## Figures to produce

### Figure 1: Study design

```text
OpenAlex full record
        ↓
hidden truth store
        ↓
controlled metadata views
        ↓
OpenAlex MCP agent arms
        ↓
mechanical adjudication
        ↓
grounding, misattribution, cost, visibility
```

### Figure 2: Nexus Score facets

```text
Provenance | People | Organizations | Funding | Access
```

### Figure 3: Metadata restoration diagonal

Rows are task classes. Columns are restored facets. Cells show improvement in correct grounding.

This should be the central validation figure for Nexus Score.

### Figure 4: Failure taxonomy

Show the split among correct answers, wrong real answers, misattributions, hallucinations, unsupported answers, refusals, and no-results.

### Figure 5: Compute cannot replace substrate

Plot cost against correct grounded answer rate.

Expected pattern:

```text
metadata-restored MCP is cheaper and more accurate than metadata-missing high-compute recovery
```

### Figure 6: Matthew-effect visibility curve

Plot Nexus Score decile against visibility lift.

Expected pattern:

```text
low-Nexus records underrepresented
high-Nexus records overrepresented
```

### Figure 7: Iterated discovery amplification

Show the Nexus Score distribution at corpus construction, first retrieval, related-paper expansion, and final generated literature map.

Expected pattern:

```text
AI-mediated discovery shifts the visible literature toward high-Nexus records across rounds
```

---

## Claims this design can support

If the results follow the expected pattern, the study can support the following claims.

Metadata depth causally affects AI grounding because the same works and same questions produce different outcomes when specific metadata facets are withheld or restored.

Nexus Score measures AI-readiness because its facet scores predict the corresponding task outcomes, while metadata-independent negative controls show little or no effect.

RAG over scholarly records can fail by misattribution rather than only by hallucination. A real paper and a real entity are not enough; the scholarly edge between them must also be true.

LLM training-data priors, web search, and higher compute do not reliably substitute for missing authoritative metadata within the tested regimes. They can increase cost and attempts, but without a verifiable edge they cannot guarantee accountability.

Open metadata repair is more sustainable than per-query agentic recovery because a repaired record becomes reusable public infrastructure.

AI-mediated discovery can amplify a Matthew effect: high-Nexus records become easier to retrieve, verify, cite, and synthesize, while low-Nexus records become less visible even when relevant.

Metadata correction is therefore not clerical cleanup. It is infrastructure for groundedness, accountability, sustainability, transparency, and equity in AI-mediated research.

---

## Limitations to state clearly

OpenAlex is used as the hidden truth store, but OpenAlex itself is not perfect. Its metadata can be incomplete or wrong. The study measures readiness relative to the chosen open substrate and should include validation checks for a sample of ground-truth edges.

The no-embedding design is intentional for speed and interpretability, but it does not test whether dense retrieval changes the magnitude of the effects. This can be a future extension.

The metadata-restoration design isolates facet effects better than provider comparison, but facets may still interact. People and Organizations metadata, for example, often work together in author-affiliation disambiguation.

The Matthew-effect analysis is about AI-mediated visibility under the tested retrieval and generation protocols. It should not be overclaimed as a universal law of scholarly visibility without cross-model and cross-domain replication.

The compute-substitution experiment should not claim that LLMs can never recover missing metadata. It should claim that recovery is less reliable, less auditable, and less reusable than substrate-level repair under the tested conditions.

---

## Final methodological statement

This study validates Nexus Score without comparing metadata providers. It uses OpenAlex as a single open scholarly knowledge graph and creates controlled metadata views of the same records. By withholding and restoring specific facets, the study tests whether metadata depth itself determines AI grounding, misattribution, refusal, cost, and visibility. The strongest evidence is a facet-specific diagonal: People metadata repairs author tasks, Organizations metadata repairs institution tasks, Funding metadata repairs funder tasks, Provenance metadata repairs citation and source tracing, and Access metadata repairs inspectable-answer tasks, while known-item negative controls remain mostly unchanged.

The accountability conclusion is that a model cannot reliably attribute what the record does not identify. The sustainability conclusion is that substrate repair amortizes while agent recovery repeats. The equity conclusion is that metadata gaps become visibility gaps in AI-mediated discovery. The Matthew-effect conclusion is that high-Nexus records can become more visible because they are safer and easier for AI systems to ground, while low-Nexus records risk being excluded from the machine-visible scholarly record.

Nexus Score is therefore a repair map for open scholarly infrastructure. It measures where the accountability layer exists, where it is missing, and what fails when AI systems are asked to reason without it.
