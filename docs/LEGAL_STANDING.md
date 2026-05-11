# Legal Standing Documentation

**Copyright © 2026 Stellar Aqua LLC. All rights reserved.**

---

This document summarizes the legal standing formalization implemented across the Sundog repository.

## What Was Done

### 1. Copyright Inventory and Documentation

**File:** `COPYRIGHT.md`

A comprehensive copyright inventory document was created that:
- Catalogs all copyrightable works into 8 buckets (website text, documentation, source code, visual assets, videos, application writeups, experimental results, contractor works)
- Clarifies what copyright does and does NOT protect (theorems, methods, algorithms, brand names are not protected by copyright)
- Establishes contractor requirements: written IP assignment agreements required BEFORE work begins
- Documents the future entity transition path

### 2. License File

**File:** `LICENSE`

MIT License with:
- Copyright holder: Stellar Aqua LLC
- Standard MIT permissions for use, modification, distribution
- Sponsorship statement included

### 3. Website Footer Updates

**Files:** public root HTML pages and the browsable docs index.

All public-facing HTML pages now include the recommended copyright notice:

```
© 2026 Stellar Aqua LLC. Sundog Research Lab is an independent applied
research program sponsored by Stellar Aqua LLC.
```

### 4. About Page Creation

**File:** `about.html`

The About page carries the Sundog Research Lab framing that:
- Positions Sundog as an independent applied research lab for systems that act without full sight
- Explains the field origin and practical sponsor relationship
- Details research posture and evidence tiers
- Clarifies what Sundog does NOT claim, including SunDog: Frozen Legacy, crypto, and universal-proof confusion
- Includes information about Stellar Aqua LLC as current sponsor and copyright holder

### 5. Navigation Updates

**Files:** All HTML files

Navigation menus updated to include About page as first link after Home/brand, establishing it as primary identity page while Origin becomes a historical/discovery story.

### 6. Repository Documentation

**Files:** `README.md`, `docs/README.md`

Copyright headers added to main repository documentation files.

### 7. Package Metadata

**File:** `package.json`

Updated with:
- Description including Sundog Research Lab and Stellar Aqua sponsorship
- Author: Stellar Aqua LLC
- License: MIT
- Repository and homepage URLs

## What This Solves

### Entity Risk Firewall

The primary goal was to create separation between:
- **Stellar Aqua LLC**: Operating company with installation liability, vendor disputes, employment issues, tax issues
- **Sundog Research Lab**: Independent research program with theorem, research IP, brand identity

This structure allows for future entity formation (Sundog Research Lab LLC, nonprofit, or IP holding entity) without pretending that separation already exists.

### Academic Credibility

The "independent applied research program sponsored by Stellar Aqua LLC" framing:
- Acknowledges the controls company origin honestly
- Positions it as an advantage (field procedure, not ivory-tower abstraction)
- Maintains research discipline through evidence tiers
- Avoids pretending to be a university lab or pure software company

### Copyright Clarity

The COPYRIGHT.md inventory provides:
- Clear ownership records for copyright registration
- Contractor work requirements to prevent IP ownership gaps
- Documentation for future entity transition and IP assignment

## Next Steps Recommended

### 1. Copyright Registration (Priority)

Register works with the U.S. Copyright Office in batches:

**Batch 1: Core Research Materials**
- Website text and copy as literary work
- Documentation (papers, guides, roadmaps) as literary work
- **Use eCO online registration:** https://www.copyright.gov/eco/

**Batch 2: Software**
- Source code as computer program
- Submit deposit copy of key modules

**Batch 3: Visual Works**
- Generated plots and charts
- Diagrams and illustrations
- Final logo/visual marks when created

Cost: ~$65 per registration
Timeline: 3-8 months for processing
Benefit: Public record, required before filing infringement suit for U.S. works

### 2. Trademark Search and Filing

**Search first:**
- "Sundog Research Lab"
- "Sundog Alignment Theorem" (may be descriptive)
- Visual marks/logos

**File if clear:**
- Use USPTO's Trademark Electronic Application System (TEAS)
- Cost: $250-$350 per class
- Timeline: 6-12 months

### 3. Contractor Agreement Template

Create a standard agreement for all contractor/freelancer work that includes:
- Work-for-hire language where applicable
- Explicit copyright assignment to Stellar Aqua LLC
- Derivative rights assignment
- Confidentiality provisions
- Payment tied to deliverables and IP assignment

**DO NOT use generic contracts without IP assignment language.**

### 4. IP Assignment Paper Trail

For any work already commissioned:
- Document what was paid for and when
- Obtain retroactive IP assignment if missing
- Create a ledger of contractor works and assignment status

### 5. Future Entity Formation

When justified (serious external momentum, funding, partnerships):

1. Form appropriate entity (LLC, nonprofit, foundation)
2. Draft IP assignment agreement from Stellar Aqua LLC to new entity
3. Transfer copyrights, trademarks, domain names
4. Stellar Aqua retains sponsor credit and appropriate license-back
5. Update all materials to reflect new ownership

### 6. Site Metadata (Already Done)

✓ Footer with copyright notice
✓ About page with entity relationships
✓ LICENSE file
✓ COPYRIGHT.md inventory
✓ Package.json metadata

## Contact for IP Questions

For copyright, trademark, licensing, or IP assignment questions regarding Sundog materials, contact Stellar Aqua LLC through the repository maintainer.

---

## References

- [U.S. Copyright Office](https://www.copyright.gov/)
- [Copyright Registration Portal (eCO)](https://www.copyright.gov/eco/)
- [USPTO Trademark Search](https://www.uspto.gov/trademarks/search)
- [TEAS Trademark Application](https://www.uspto.gov/trademarks/apply)

---

**Last Updated:** May 11, 2026
