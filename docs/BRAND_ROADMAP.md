# Sundog Brand and IP Roadmap

**Copyright © 2026 Stellar Aqua LLC. All rights reserved.**

---

This roadmap formalizes the brand, copyright, and entity-formation plan for Sundog Research Lab. It progresses from immediate legal hygiene (Phase 0) through formal registration (Phase 1) to eventual entity independence (Phase 2).

For public voice, Mythos stress-test lessons, and the About-page brand spine,
see [`BRAND_POSITIONING.md`](BRAND_POSITIONING.md). This document remains the
legal/IP execution roadmap.

## Current State: Foundation Complete

**Status:** ✅ Complete (May 2026)

The repository now has:
- ✅ COPYRIGHT.md inventory of all copyrightable works (8 buckets)
- ✅ LICENSE file (MIT with Stellar Aqua LLC copyright holder)
- ✅ Footer copyright notices across public HTML pages
- ✅ About page establishing Sundog Research Lab's public research posture and current Stellar Aqua LLC sponsorship structure
- ✅ Brand positioning note separating the Mythos stress-test lesson from the legal/IP roadmap
- ✅ Gemini benchmark filed as second brand stress test, with crawler/redirect debugging topics recorded in `BRAND_POSITIONING.md`
- ✅ Updated navigation with About page as primary identity page
- ✅ Copyright headers on README.md and major documentation files
- ✅ Package.json metadata (author, license, description)

**Firewall established:** Sundog Research Lab exists as public-facing research program identity, distinct from Stellar Aqua's operating company liability, while Stellar Aqua retains current copyright ownership and sponsor role.

---

## Phase 0: Legal Hygiene and Documentation

**Objective:** Ensure all existing IP is documented, assigned, and protected with minimal formality before external engagement.

**Timeline:** Immediate (0-2 months)

**Priority:** High - Required before accepting outside contributions, contractor work, or publication

### 0.1 Contractor Agreement Template

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Draft standard contractor agreement with explicit IP assignment language
- [ ] Include work-made-for-hire provisions where applicable
- [ ] Add copyright and derivative rights assignment to Stellar Aqua LLC
- [ ] Add confidentiality and non-compete clauses appropriate for research work
- [ ] Tie payment milestones to deliverable acceptance and IP assignment
- [ ] Review with attorney if budget permits

**Deliverable:** `docs/contracts/CONTRACTOR_AGREEMENT_TEMPLATE.md` or `.docx`

**Critical:** Independent contractor work is NOT automatically owned by the hiring party under U.S. copyright law. Written assignment is legally required.

**Cost:** $0-500 (DIY to attorney review)

---

### 0.2 IP Assignment Paper Trail

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Audit all paid contractor/freelancer/agent work to date
- [ ] Document what was commissioned, when, and what was delivered
- [ ] Identify gaps: work without written IP assignment
- [ ] Obtain retroactive IP assignment agreements where possible
- [ ] Create ledger: `docs/contracts/IP_ASSIGNMENT_LEDGER.md`

**Ledger Format:**
```
| Work Description | Contractor | Date | Payment | Assignment Status | Assignment Date | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Logo design | Jane Designer | 2026-03 | $500 | ✅ Signed | 2026-03-15 | Full rights assigned |
| Video production | VideoCo | 2026-04 | $2000 | ⚠️ Pending | - | Retroactive needed |
```

**Deliverable:** Complete IP assignment ledger with no gaps

**Cost:** $0 (documentation only, unless legal review needed)

---

### 0.3 Third-Party Reuse Documentation

**Status:** ✅ Complete (see `docs/THIRD_PARTY_REUSE.md`)

**Tasks:**
- [x] Document all third-party code, libraries, and dependencies
- [x] Verify license compatibility with MIT
- [x] Maintain reuse ledger for MuJoCo, RK4 implementations, etc.

**Deliverable:** Already exists at `docs/THIRD_PARTY_REUSE.md`

---

### 0.4 Contribution Guidelines

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Create CONTRIBUTING.md with IP assignment requirements
- [ ] Require CLA (Contributor License Agreement) for external contributions
- [ ] Document process for accepting pull requests with IP clean room
- [ ] Add IP assignment language to PR templates

**Deliverable:** `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md` updates

**Cost:** $0

**Urgency:** Medium - Required before accepting external code contributions

---

## Phase 1: Formal Registration and Protection

**Objective:** Register copyrights and trademarks to establish public record and legal standing.

**Timeline:** 2-12 months from Phase 0 completion

**Priority:** Medium - Recommended before major publication, product launch, or fundraising

### 1.1 Copyright Registration (Priority)

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Register Batch 1: Core Research Materials
  - [ ] Website text and copy as literary work
  - [ ] Documentation (papers, guides, roadmaps) as literary work
  - [ ] Use eCO online registration: https://www.copyright.gov/eco/
- [ ] Register Batch 2: Software
  - [ ] Source code as computer program
  - [ ] Submit deposit copy of key modules (agents, experiments, env)
- [ ] Register Batch 3: Visual Works
  - [ ] Generated plots and charts
  - [ ] Diagrams and illustrations
  - [ ] Final logo/visual marks when created

**Cost:** ~$65 per registration × 3 batches = $195 minimum

**Timeline:** 3-8 months for processing after submission

**Benefit:**
- Public record of copyright ownership
- Required before filing infringement suit for U.S. works
- Enables statutory damages in infringement cases
- Professional credibility for research program

**Deliverable:** Copyright registration certificates for all batches

**Process:**
1. Prepare deposit copies of works
2. Complete Form CO via eCO system
3. Pay fees
4. Track application status
5. Store certificates in `docs/certificates/`

---

### 1.2 Trademark Search and Filing

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Conduct USPTO trademark search for "Sundog Research Lab"
- [ ] Conduct search for "Sundog Alignment Theorem" (may be descriptive)
- [ ] Search visual marks/logos once finalized
- [ ] File trademark application if search is clear
  - [ ] Use USPTO TEAS (Trademark Electronic Application System)
  - [ ] File in Class 42 (Scientific and technological services)
  - [ ] Possibly Class 9 (Software) and Class 41 (Education)
- [ ] Respond to USPTO office actions
- [ ] Monitor opposition period
- [ ] Obtain registration certificate

**Cost:** $250-$350 per class × 1-3 classes = $250-1,050

**Timeline:** 6-12 months from filing to registration

**Benefit:**
- Protects brand name "Sundog Research Lab"
- Copyright does NOT protect brand names; trademark is required
- Prevents confusingly similar uses by others
- Enables ® symbol after registration

**Deliverable:** Trademark registration certificate(s)

**Critical Decision:** Determine if "Sundog Alignment Theorem" qualifies as trademark or is merely descriptive. Likely better protected through publication and citation than trademark.

---

### 1.3 Domain Name Portfolio

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Register primary domains if not already owned:
  - [ ] sundogresearch.com
  - [ ] sundogresearchlab.com
  - [ ] sundog.cc (already owned?)
- [ ] Register defensive variations:
  - [ ] sundogai.com
  - [ ] sundogsystems.com
- [ ] Point domains to canonical site or holding pages
- [ ] Document ownership in `docs/BRAND_ASSETS.md`

**Cost:** $10-20 per domain per year

**Timeline:** Immediate

**Benefit:** Prevents domain squatting and brand confusion

---

### 1.4 Logo and Visual Identity Lock

**Status:** ⚠️ Pending

**Tasks:**
- [ ] Finalize Sundog Research Lab logo (if not already final)
- [ ] Create brand guidelines: colors, typography, usage rules
- [ ] Document in `docs/BRAND_GUIDELINES.md`
- [ ] Register logo as visual trademark (see 1.2)
- [ ] Ensure logo has explicit IP assignment from designer (see 0.2)

**Cost:** $0-$1000 (depends on whether designer is needed)

**Timeline:** 1-2 months

**Deliverable:** Final logo files, brand guidelines, trademark filing

---

## Phase 2: Entity Formation and Independence

**Objective:** Form separate Sundog Research Lab entity and transfer IP out of Stellar Aqua LLC to establish research program as independent legal entity.

**Timeline:** 12-24+ months from now, triggered by external momentum

**Priority:** Low - Only pursue when justified by serious external engagement

**Triggers for Phase 2:**
- Significant outside funding or grant opportunity
- Partnership or collaboration requiring separate entity
- Product revenue justifying corporate structure
- Desire to limit Stellar Aqua operating liability exposure
- Academic or institutional affiliation requiring nonprofit structure

### 2.1 Entity Formation Decision

**Status:** ⚠️ Pending - Not yet justified

**Options:**

#### Option A: Sundog Research Lab LLC (For-Profit)
- **Best for:** Product development, commercial partnerships, investor funding
- **Cost:** $100-500 filing + $800-1500/year franchise tax (CA) or $0-300/year (DE/WY)
- **Pros:** Flexible, investor-ready, straightforward IP transfer
- **Cons:** Taxed as business, no grant eligibility

#### Option B: Sundog Research Foundation (501(c)(3) Nonprofit)
- **Best for:** Grant funding, academic partnerships, pure research mission
- **Cost:** $275-600 IRS filing + state filing + annual compliance
- **Pros:** Tax-exempt, grant-eligible, mission credibility
- **Cons:** Slower formation, restricted activities, must serve public benefit

#### Option C: Fiscal Sponsorship
- **Best for:** Near-term grant opportunity without full entity formation
- **Cost:** 5-15% of grant funds
- **Pros:** Fast, no entity formation, nonprofit benefits
- **Cons:** Sponsor controls funds, temporary solution

**Decision Criteria:**
- Revenue model: product sales vs. grants
- Control requirements: founder control vs. board governance
- Timeline: immediate need vs. long-term plan
- Tax implications: profit distribution vs. tax exemption

**Deliverable:** Entity formation decision memo

---

### 2.2 IP Assignment Agreement

**Status:** ⚠️ Pending - Blocked by 2.1

**Tasks:**
- [ ] Draft IP assignment agreement from Stellar Aqua LLC to new entity
- [ ] Include all copyrights, trademarks, domain names, and related IP
- [ ] Stellar Aqua retains:
  - [ ] Historical sponsor credit and attribution
  - [ ] License-back rights to use Sundog materials for business purposes
  - [ ] Right to reference Sundog origin story and field discovery
- [ ] New entity receives:
  - [ ] All copyright registrations (transfer with USPTO)
  - [ ] All trademark registrations (transfer with USPTO)
  - [ ] Domain name transfers
  - [ ] Physical assets (if any)
  - [ ] Contributor agreements and assignment ledger

**Cost:** $500-2000 (attorney review strongly recommended)

**Timeline:** 1-2 months

**Deliverable:** Executed IP assignment agreement

**Critical:** This is a legal transfer of ownership. Attorney review is not optional.

---

### 2.3 Update All Public Materials

**Status:** ⚠️ Pending - Blocked by 2.2

**Tasks:**
- [ ] Update footer copyright notice to new entity:
  ```
  © 2026 Sundog Research Lab LLC. Initial research and application
  development sponsored by Stellar Aqua LLC.
  ```
- [ ] Update About page to reflect new ownership structure
- [ ] Update LICENSE file with new copyright holder
- [ ] Update COPYRIGHT.md with transfer date and new owner
- [ ] Update package.json author and metadata
- [ ] File copyright transfer notices with Copyright Office
- [ ] File trademark transfer documents with USPTO
- [ ] Transfer domain registrations to new entity
- [ ] Update GitHub repository ownership or transfer repository

**Cost:** $0 (just updates)

**Timeline:** 1-2 weeks

**Deliverable:** All materials reflect new entity ownership while preserving Stellar Aqua historical credit

---

### 2.4 Ongoing Entity Operations

**Status:** ⚠️ Pending - Blocked by 2.1

**Tasks:**
- [ ] Obtain EIN (Employer Identification Number) for new entity
- [ ] Open business bank account
- [ ] File annual reports (LLC) or Form 990 (nonprofit)
- [ ] Maintain registered agent and good standing
- [ ] Renew trademark registrations (every 10 years)
- [ ] Maintain copyright registrations (lifetime + 70 years)
- [ ] Update contractor agreements to assign IP to new entity
- [ ] Maintain CLA process for external contributions

**Cost:** $100-1000/year depending on entity type and state

**Timeline:** Ongoing after formation

**Deliverable:** Compliant entity in good standing

---

## Cost Summary

| Phase | Item | Cost | When |
| --- | --- | --- | --- |
| **Phase 0** | Contractor agreement template | $0-500 | Immediate |
| **Phase 0** | IP assignment audit | $0 | Immediate |
| **Phase 0** | CONTRIBUTING.md | $0 | Before external contributions |
| **Phase 1** | Copyright registration (3 batches) | $195 | Before major publication |
| **Phase 1** | Trademark registration (1-3 classes) | $250-1,050 | Before product launch |
| **Phase 1** | Domain names (5-10 domains) | $50-200/year | Immediate |
| **Phase 1** | Logo/brand finalization | $0-1,000 | Before trademark filing |
| **Phase 2** | Entity formation | $375-1,500 | When externally justified |
| **Phase 2** | IP assignment agreement (attorney) | $500-2,000 | With entity formation |
| **Phase 2** | Annual entity costs | $100-1,500/year | Ongoing after formation |
| **Total Phase 0-1** | **$495-2,950** | **0-12 months** |
| **Total Phase 2** | **$975-5,000 + annual** | **12-24+ months** |

---

## Success Criteria

### Phase 0 Complete When:
- ✅ All contractor work has written IP assignment
- ✅ Contractor agreement template exists
- ✅ CONTRIBUTING.md prevents IP gaps in external contributions
- ✅ IP assignment ledger shows no gaps

### Phase 1 Complete When:
- ✅ Copyright registrations filed and certificates received
- ✅ Trademark search completed and applications filed
- ✅ Domain portfolio secured
- ✅ Logo and brand identity finalized and documented

### Phase 2 Complete When:
- ✅ New entity formed and in good standing
- ✅ All IP transferred to new entity with attorney-reviewed agreements
- ✅ All public materials updated to reflect new ownership
- ✅ Stellar Aqua retains appropriate sponsor credit and license-back
- ✅ New entity has operational bank account and EIN
- ✅ Ongoing compliance processes established

---

## Decision Gates

**Proceed to Phase 1 when:**
- Phase 0 contractor audit shows clean IP chain
- Ready for external publication or product launch
- Budget available for registration fees

**Proceed to Phase 2 when:**
- External funding, partnership, or collaboration opportunity requires separate entity
- Product revenue justifies corporate structure
- Desire to limit Stellar Aqua operating liability is urgent
- Academic or institutional partnership requires nonprofit structure

**Do NOT proceed to Phase 2 until:**
- Phase 0 and Phase 1 are complete
- Attorney consulted on entity type and IP transfer
- Cash reserves available for formation and 1 year operating costs
- Clear business justification exists (not just "nice to have")

---

## Maintenance and Updates

This roadmap should be reviewed and updated:
- After completing each phase
- Every 6 months during Phase 0-1
- Annually after Phase 2
- When external circumstances change (funding, partnerships, legal landscape)

**Document Owner:** Repository maintainer

**Last Updated:** May 11, 2026

---

## References

- [U.S. Copyright Office](https://www.copyright.gov/)
- [Copyright Registration Portal (eCO)](https://www.copyright.gov/eco/)
- [USPTO Trademark Search](https://www.uspto.gov/trademarks/search)
- [TEAS Trademark Application](https://www.uspto.gov/trademarks/apply)
- [IRS Form 1023 (501c3 Application)](https://www.irs.gov/forms-pubs/about-form-1023)
- [Small Business Administration: Business Structure](https://www.sba.gov/business-guide/launch-your-business/choose-business-structure)
