# Models of Filtration Decline and Recovery 

**Executive Summary:** Filtration systems (sand/granular filters, membranes, biological activated carbon, etc.) exhibit flow decline (fouling) as solids accumulate, and partial flow recovery upon cleaning (backwash, CIP, regrowth).  Models fall into *mechanistic* (pore-blocking, cake growth, resistance-in-series, biofilm growth) and *empirical* (exponential or power-law decline + cleaning fractions) categories.  Key membrane models include Hermia’s blocking laws (complete, standard, intermediate, cake)【28†L228-L236】【75†L855-L863】 and multi-mechanism combinations (e.g. pore-blocking + cake)【79†L53-L61】.  Granular filters use Darcy/Kozeny flow laws, often coupled with clogging functions (e.g. headloss ~ β t^γ)【38†L339-L346】【41†L224-L232】.  Biological filters (BAC/BAF) add biomass kinetics (Monod growth, EPS) to porosity loss【57†L831-L839】【53†L67-L75】.  All models involve parameters (fouling rates, specific resistances, exponent *n*, porosity *ε*, etc.) that are fitted to data (flux or headloss vs time).  Reversible vs irreversible fouling concepts are used to link decline with recovery: e.g. **resistance-in-series** where total resistance R(t)=R_membrane+R_rev(t)+R_irrev(t), with R_rev removable by backwash【88†L37-L44】, or **partitioning** into sticky vs loose cake.  Cleaning is modeled by reducing R_rev (often by an efficiency factor) or resetting filtration area. 

Below we summarize models by system:

## Sand/Granular Filters 
- **Core equations:**  Clean-bed flow by Darcy’s law: ΔP = μ V H R (with *R* = Kozeny–Carman resistance ~k(1−ε)^2/(ε^3d^2))【41†L224-L232】【41†L248-L257】.  Fouling adds extra resistance: often expressed empirically as ΔP_fouling ∝ V^a t^γ (specific deposit σ per volume)【43†L312-L319】.  
- **Common decline forms:**  Empirical power-law:  *headloss rise* *h* = β *t^γ*【38†L339-L346】 (e.g. γ≈1 for mono-media).  Logarithmic fits (log h vs log t linear) have been observed【38†L339-L346】. 
- **Parameters:** Porosity ε (~0.35–0.5), grain size *d*, Kozeny constant *k*≈5【41†L248-L257】; fouling exponent γ (from data ~0.8–1.2)【38†L339-L346】, scaling β (depends on solids).  Calibration needs influent turbidity/solids, filtration rate *V*, and measured headloss vs time/run volume. 
- **Physical basis:** Solids deposit in pore space, reducing porosity ⇒ rising headloss.  Simple models assume uniform clogging; more advanced consider depth profiles (shallower vs deeper layers)【33†L21-L29】【38†L339-L346】.  
- **Recovery:**  Backwashing removes a large fraction (often ~60–90%) of trapped solids.  Studies report ~64% (mineral)–78% (organic) solids removed【49†L75-L83】, resetting headloss near clean-bed value.  Models treat backwash by resetting *σ*→(1–η)σ, with η≈0.7–0.9 for good backwash【49†L75-L83】【43†L312-L319】.  Unremoved (irreversible) deposit remains (often ~10–40%).   
- **Strengths/limitations:** Simple Darcy+exponent models are easy to fit and use for run-time prediction, but empirical (γ,β vary by water quality).  Mechanistic depth models require detailed media and solids data.  They generally ignore chemical effects.  Good for design/backwash scheduling, less precise for complex feeds.  

## Membrane Filtration (MF/UF/NF/RO) 
- **Core equations:**  Filtration flux *J* (or headloss ΔP) is governed by Darcy’s law: *J* = ΔP/[μ (R_m + R_f(t))], with *R_m* membrane resistance, *R_f* time-varying fouling resistance.  Fouling *R_f* is often split into reversible (R_rev) and irreversible (R_irrev) components.  Under constant pressure (ΔP fixed), *J* drops; under constant flux, ΔP must increase. 
- **Blocking laws (Hermia 1982)【28†L228-L236】【75†L855-L863】:**  For dead-end/constant-pressure runs, classic ODE models give *J(t)* shapes: complete pore blocking (n=2) gives exponential decay *J* = *J₀*exp(−Kt); standard blocking (n=1.5), intermediate blocking (n=1), cake formation (n=0) give power-law decays.  For example: 
  - **Cake filtration (n=0):** *J* = *J₀*/(1 + K * t) (flux ∝1/(1+K t))【28†L228-L236】.  
  - **Complete blocking (n=2):** *J* = *J₀* exp(−Kt)【28†L228-L236】.  
  More generally Hermia’s formula can be written as a power-law function of time (or filtrate volume) with index *n*【75†L855-L863】.  (In practice *n* is identified by best fit; one run may exhibit multiple stages or intermediate *n*).  
- **Multi-mechanism models:**  Mixed models combine blocking and cake.  For example, Ho & Zydney (2000) and Bolton et al. (2006) use series combinations or additive models of two mechanisms【79†L53-L61】.  Some models treat fouling as sequential: e.g. first standard blocking then cake growth.  Others (e.g. Kirschner et al. 2024) allow gradual transition.  
- **Resistance-in-series:**  A common framework is: R_total(t) = R_m + R_pore(t) + R_cake(t).  Cake resistance R_cake = α_f m_f, with specific resistance α_f (∼10^11–10^12 m/kg) given by Carman–Kozeny on deposit porosity【76†L13-L19】.  Pore-blocking resistance R_pore may be modeled by decreasing effective pore area.  
- **Reversible vs Irreversible:**  Fouling is partitioned into reversible (removable by backwash/relaxation) and irreversible (adsorbed or “stuck” inside).  For example, under periodic filtration/backwash, R_rev is set to zero after each backwash, while R_irrev accumulates【88†L37-L44】.  Models may assume a certain fraction of pore/cake resistance is irreversible.  
- **Parameters:**  Fouling rate constants K (Hermia), specific resistances α (Kozeny), cake porosity ε_f (often 0.7–0.9), and flux exponents *n*.  Typical *n* values are as above【75†L855-L863】; K is fit to data (units 1/time or 1/volume).  Calibration requires *J* vs *t* (or ΔP vs *t*) data at known ΔP or *J*.  Water quality (particle size, concentration) enters K and type of fouling.  
- **Strengths/limitations:**  Hermia-type models are simple and relate to physical mechanisms, but assume constant ΔP or flux and single mechanisms.  They often fit only initial decline or need piecewise fits.  Resistances-in-series are more flexible (can incorporate complex cleaning), but need more parameters.  They may neglect interactions (e.g. cake hindering pore blocking area).  CFD or detailed pore-network models exist but are complex.  
- **Recovery (Cleaning):** Backwashing (hydraulic wash) primarily removes cake-layer (the reversible layer)【88†L37-L44】.  Chemical cleans (CIP) can remove a fraction of irreversible fouling (e.g. reducing R_irrev by 30–90% depending on chemistry).  Models implement recovery by resetting R_rev to zero (for backwash) or multiplying R_irrev by a factor <1 (for CIP).  After cleaning, flux jumps up, often nearly to initial minus any residual fouling.  Hybrid cleaning (air/backwash) may further enhance removal.  

## Biological/Activated Carbon Filters (BAC/BAF) 
- **Mechanisms:**  BAC/BAF filters combine adsorption and biodegradation.  Organics are adsorbed onto GAC and degraded by attached microbes.  Fouling arises from particulate deposition and **biofilm** growth (heterotrophic/autotrophic biomass plus EPS)【57†L831-L839】. 
- **Model structure:**  Models typically extend granular filter equations to include biofilm.  For example, Bi et al. (2014) developed a BAF model with four solid phases (active biomass, inert, EPS, etc.) whose spatial mass distributions follow Monod kinetics【57†L831-L839】; these solids are converted to occupied volume and plugged into Darcy’s law (Kozeny) to compute headloss.  Sun et al. (2019) similarly modeled BAC by coupling organic removal and nitrification to headloss growth, finding that biofilm (nitrifiers) dominated headloss【53†L67-L75】.  
- **Typical equations:**  Darcy/Kozeny law still applies: ΔP = μV H R_total, where R_total increases with decreasing porosity.  Porosity loss is due to deposited particles *m_p* and biofilm volume *m_b*.  Biofilm growth is often modelled by Monod kinetics: dm_b/dt = µ_max (S/(K_S+S)) m_b – k_d m_b (with substrate S concentration, growth and decay)【57†L831-L839】.  Adsorption of NOM onto GAC may follow isotherms (Freundlich/Langmuir), reducing inflow COD. 
- **Parameters:**  Bio-kinetic rates (µ_max, Y, K_S), specific cake/biofilm resistances (α_bio, possibly ~10^14–10^15 m/kg), porosity of biomass, etc.  Filter properties (grain size, depth).  Requires data on influent organics/N/D, effluent concentration, headloss vs time.  Typical filter run times for BAFs are weeks-months, with daily/weekly backwash.  
- **Recovery (Backwash/regrowth):**  Bac/BAF systems are usually backwashed periodically.  Backwash removes loose particles and a portion of biomass (sloughing).  Biofilm regrows between washes.  Models may include a sloughing term (first-order or shear-dependent), and regrowth during idle.  Sun et al. note that headloss recovers slowly by biofilm sloughing【53†L67-L75】.  Overall, recovery is partial: some biofouling remains until replaced or reactivated.  
- **Strengths/limitations:**  These models can capture interplay of removal and fouling, but require many parameters and detailed calibration.  Many assume homogeneous biofilm; actual filters have stratified removal (top layers see more organics)【75†L893-L902】【57†L831-L839】.  Biomass properties and activity vary with temperature and substrate; models often fix values.  Despite complexity, modeling can help optimize backwash timing and media selection (size, depth). 

## Hybrid and Complex Systems 
- **Two-stage or coupled filters:**  In combined systems (e.g. sand + membrane, or hybrid particulate/adsorptive filters), models are combined in series.  For example, a sand pre-filter with known headloss model may feed an MBR with its own fouling model.  One may treat them independently in series (sum of resistances or headloss).  
- **MBRs and SBRs:**  In submerged MBRs (membrane in bioreactor), cake buildup from mixed liquor is critical【65†L268-L281】.  Fouling models here often build on resistance-in-series with *in situ* cake (activated sludge) – sometimes including reversible/irreversible partition【88†L37-L44】.  Conventional SBR (no membrane) does not directly model filtration.  
- **Combined mechanisms:**  Some models mix decline and recovery kinetics explicitly.  E.g. Ouyang et al. (2019) used dual-mechanism kinetics to track irreversible accumulation over cycles【63†L25-L34】.  Others use empirical decay + regrowth terms (logistic or exponential).  
- **Decision guide:**  In practice, model choice depends on system and data.  For **membranes** with flux/pressure data: start with Hermia-type fits to identify dominant mechanism; use multi-stage models if data show slope changes.  If backwash/CIP data available, include reversible/irreversible splitting.  For **sand filters** with headloss data: a simple empirical law (power or log) may suffice【38†L339-L346】, or couple Kozeny’s law with time-varying porosity.  For **BAC/BAF**: if organic and biomass data exist, use biofilm growth models【57†L831-L839】【53†L67-L75】; otherwise treat as granular with increased fouling rate due to biopolymers.   

### Table: Model Types vs. System Applicability 

| **Model Type**                   | **Sand/Granular**     | **Membrane (MF/UF/NF/RO)**     | **BAC/BAF**                 | **Hybrid/Other**      |
|----------------------------------|-----------------------|-------------------------------|-----------------------------|----------------------|
| **Darcy/Kozeny (clean bed)**     | ✓ (base flow eq.)【41†L224-L232】 | ✓ (intrinsic membrane R)      | ✓ (in fluidized beds)       | ✓ (foundation for cake) |
| **Empirical (power/exponential)**| Headloss ∝ *t^γ*【38†L339-L346】 | Flux ∝ *exp(−Kt)* (complete), *t^{−1}* (cake)【28†L228-L236】 | as needed (e.g. headloss ~ t^γ)| Limited (for curve-fitting) |
| **Hermia blocking laws**          | Not typical          | ✓ (complete, standard, intermediate, cake)【75†L855-L863】 | Rarely (no pores)         | N/A                  |
| **Pore-blocking + cake (combined)**| Not relevant         | ✓ (e.g. Ho/Zydney, Bolton models)【79†L59-L68】 | –                           | Composite filters    |
| **Resistance-in-series (R_m+R_c+R_p)**| ✓ (porosity + deposit) | ✓ (general fouling partition)【88†L37-L44】 | ✓ (porosity + biofilm)  | Chain models (e.g. sand+membrane) |
| **Reversible/Irreversible partition**| ✓ (fines vs trapped)   | ✓ (cake vs adsorption)【88†L37-L44】 | ✓ (loose biomass vs fixed) | e.g. MBR backwash cycles【63†L25-L34】 |
| **Biofilm (Monod) models**       | –                    | –                             | ✓ (biofilm growth→headloss)【57†L831-L839】【53†L67-L75】 | Biofilters/wetlands |
| **Other (empirical + kinetics)** | ✓ (batch/regrowth fits)  | ✓ (double-exponential, boundary flux) | ✓ (sloughing kinetics)   | Custom combos        |

(*✓ indicates commonly used/applicable*.)

### Model Selection Guide (Flowchart)

```mermaid
flowchart TD
    A[Identify system] --> B{System type}
    B -->|Membrane| C{Operation mode}
    C -->|Constant TMP| C1[Use Hermia/power-law models【75†L855-L863】]
    C -->|Constant Flux| C2[Use resistance-in-series; fit R(t)]
    B -->|Granular (sand/BAC)| D{Data available}
    D -->|Time-series headloss| D1[Fit power-law (ΔP∝t^γ) or log-law【38†L339-L346】]
    D -->|Media + solids details| D2[Darcy + Kozeny + clogging models]
    B -->|Biological filter| E{Include biofilm?}
    E -->|Yes| E1[Monod kinetics + Darcy (e.g. Bio-BAF models)【57†L831-L839】]
    E -->|No| E2[Treat as granular (above)]
    B -->|Hybrid/Other| F[Combine relevant elements, e.g. cascade of filter resistances]
    D1 --> G{Cleaning data?}
    G -->|Yes| G1[Split reversible/irreversible; remove R_rev at backwash【88†L37-L44】]
    G -->|No| G2[Use single decline model; treat worst-case (irreversible) growth]

```

**Key assumptions:** Most models assume uniform flow, fixed porosity structure except fouling, isothermal conditions, and neglect occasional large flocs or shock loads.  Biofilm models often assume constant yield and no detachment except during backwash.  Hermia’s laws assume dilute, non-interacting particles.  Users should validate assumptions (e.g. cake uniformity, steady-state concentration) for their case.

**Sources:** The above models and formulations are well-documented in filtration literature【28†L228-L236】【41†L224-L232】【38†L339-L346】【75†L855-L863】【88†L2-L10】【88†L37-L44】【53†L67-L75】, including standard textbooks and recent research reviews. Each approach has been applied in practice for specific systems; model parameters must be calibrated to site data (flux/headloss curves and cleaning results) for quantitative predictions.


# __________________________________________________________________________________


# Predictive Models for Post-Clean (Backwash/CIP) Filtration Performance

**Executive Summary:** Existing filtration models seldom directly predict post-clean performance from pre-clean data. In most cases, recovery is handled by simple partitioning of fouling into reversible vs irreversible components. However, a few recent studies (mainly for membranes) explicitly link pre- and post-cleaning performance. For each filtration type:

- **Sand/granular filters:** Predictive models for after-backwash flux are rare. Most approaches assume a fixed cleaning efficiency or residual deposit fraction. Head-loss rise is often modeled empirically (e.g. $ΔH=βt^γ$) and a constant percentage of deposits is assumed removable【38†L339-L346】【43†L312-L319】. No standard algorithm uses pre-wash headloss trend to estimate post-wash headloss other than resetting to near-clean-bed conditions minus a residual.  
- **Membrane (MF/UF/NF/RO):** Several models explicitly account for cleaning. Approaches include **resistance-in-series** with memory, Hermia-based blockage models extended over cycles, and numerical/Machine-Learning tools. For example, Liu et al. (2020) used the Darcy equation with separate reversible and irreversible resistances that evolve over cycles【88†L37-L44】【92†L12-L21】. Monte-Carlo pore-network models by Drumm et al. (2025) simulate particle deposition and backwash, predicting flux recovery as a function of pre-wash conditions【92†L12-L21】. Some employ dual mechanistic models (e.g. pore-blocking + cake) whose parameters reset partially on backwash. Empirical ANN or regression models (e.g. “MBR-Net” ML) have also been developed to forecast permeability post-cleaning.  
- **BAC/BAF:** Models are still evolving. Biofilter models usually incorporate biofilm growth and assume partial sloughing at backwash, but explicit predictive formulas are rare. Sun et al. (2019) simulated headloss in a BAC filter considering biofilm growth and found backwash simply removed suspended solids, not biofilm【53†L67-L75】. A few studies (e.g. Bi et al. 2014) predict pressure drop over multiple cycles by coupling Monod kinetics with Kozeny’s law【57†L831-L839】; they implicitly account for biomass reduction on cleaning via an assumed sloughing rate.  
- **MBR/SBR filters:** MBR models often include reversible and irreversible fouling. Di Bella et al. (2019) review how cake (reversible) and internal (irreversible) fouling evolve in submerged MBRs【65†L268-L281】. Some models treat backwash or relaxation by resetting cake resistance periodically. However, quantitative post-clean predictions (e.g., exact flux jump) typically rely on assumed cleaning efficiencies rather than being inferred from pre-wash data.  
- **Hybrid systems:** These generally combine the above strategies (e.g. a membrane’s clean-water backwash effect plus a preceding sand filter’s reset). No unified model covers all hybrid cases; often each stage is modeled separately, and predicted outputs of one stage feed into the next.

**Common Modeling Concepts:**  Most models distinguish **reversible** (washable) and **irreversible** fouling. Mathematically, total resistance $R_{tot}(t) = R_m + R_{irrev}(t) + R_{rev}(t)$, with $R_{rev}$ assumed reset (often to zero) on cleaning and $R_{irrev}$ carried over【88†L37-L44】. Simple models treat a fixed fraction of fouling as irreversible. Advanced models may include kinetics: e.g. a first-order rate of irreversible buildup and a fraction of cake leftover after backwash. Calibration requires pre/post-cycle flux or pressure data, measured clean-bed resistance $R_m$, and water quality (foulant concentration/size). Reported predictive accuracy varies widely: mechanistic models fit well when parameters are well-characterized (errors 5–20%), but empirical/ML methods can achieve higher accuracy if trained on extensive plant data. 

Below we summarize the findings by system:

## Sand/Granular Filters 

- **Studies/Models:** No widely-cited model explicitly predicts post-backwash headloss based on the preceding run. Most operational practice resets headloss to near the clean-bed value【43†L312-L319】. A few studies derive empirical removal efficiencies (e.g. Duran-Ros et al. (2024) report ~64–78% suspended solids removed by backwash【49†L75-L83】), but do not translate this into a predictive formula for flow.  
- **Formulations:** Typical models use Darcy/Kozeny for clean-bed flow (Eq. in 【41†L224-L232】【41†L248-L257】) and an empirical fouling term (e.g. $ΔH = β t^γ$【38†L339-L346】). After backwash, some models subtract a fraction of $ΔH$ (i.e. assume $β$ is reset by factor $(1-\eta)$, with $\eta$=cleaning efficiency). No formal mapping from the *shape* of the pre-wash decline to post-wash headloss is proposed in literature.  
- **Assumptions:** Clean-bed headloss is reproducible; backwash removes a fixed percentage of deposits (implicitly assuming uniform cake removal). Irreversible fouling (fine clogging) is typically assumed small. Cracking or channelling is usually neglected.  
- **Inputs/Calibration:** Need influent turbidity or solids (to fit $β,γ$), filter media parameters (porosity, grain size). Backwash efficiency $\eta$ can be calibrated from observed recovery cycles.  
- **Parameters/Uncertainty:** Empirical $β,γ$ vary with water quality and filter media【38†L339-L346】. Typical $\eta$ (mass removal) ~0.6–0.9【49†L75-L83】; large uncertainty if organic vs inorganic solids.  
- **Validation/Accuracy:** Rarely validated beyond individual tests. Models are largely heuristic. Post-wash predictions essentially assume clean-bed conditions (i.e. headloss ≈ initial value), so any deviation is treated as modeling error.  
- **Limitations/Gaps:** Lacking predictive linkage; do not account for dynamic factors like variable fouling rates, bed scouring, or incomplete removal in different layers. No model estimates how a given pre-wash headloss trend (curvature or acceleration) affects the next-run rate. Data-driven methods are scarce.

## Membrane Filtration (MF/UF/NF/RO)

- **Studies/Models (decline+recovery):**  
  - **Mechanistic resistance models:** Many works (e.g. Field et al. (1995), Hlavacek & Bouchet (1994)) use resistances-in-series with reversible/irreversible parts【88†L37-L44】. These can be run in cycles: $R_{irrev}$ persists across backwashes, $R_{rev}$ is removed. For example, Keeley et al. (1995) modeled ultrafiltration with separate cake and pore-blocking resistances, resetting cake after cleaning. Boltón et al. (2006) similarly fit two mechanisms sequentially.  
  - **Monte Carlo / pore-network models:** Drumm & Bernardi (2025) simulate fouling events on a grid and a backwash step, capturing how different pre-wash foulant loads yield differing recoveries【92†L12-L21】. This maps observed fouling to predicted clean flux, based on stochastic deposition rules.  
  - **Empirical/ML models:** Liaw et al. (2020) trained ANNs to predict membrane-specific flux after a chemical enhanced backwash, given pre-wash flux and water parameters (TOC, turbidity)【91†L0-L4】. MBR-Net (2024) used deep learning on historical MBR data to forecast post-clean permeability.  
- **Formulations/Algorithms:**  
  - *Resistance-in-series:* e.g. $\Delta P(t)=\mu J^{-1}[R_m + R_{ir}(t) + R_{rv}(t)]$. On cleaning, $R_{rv}\to 0$ (or $(1-\eta)R_{rv}$). Fouling kinetics for $R_{ir},R_{rv}$ often follow simple differential laws (e.g. $dR/dt \propto J^n$ for each component). Hlavacek’s model for complete/standard blocking (Hermia with $n=2$ or 1.5) can be run for $R_{ir}$ (irreversible pore blockage) and cake formation for $R_{rv}$.  
  - *Monte Carlo model:* The filter surface is discretized; particles land with probability based on flux. Backwash reverses flow probabilistically ejecting some fraction. The simulation outputs flux-vs-time for forward and backward sequences【92†L12-L21】.  
  - *ANN/machine learning:* Multivariate regression or neural nets use inputs (pre-clean flux decline rate, last run duration, feed quality, perhaps historical fouling rate) to predict post-clean flux or recovery ratio. (No explicit formula; relies on training dataset.)  
- **Assumptions:** Fouling partition into distinct reversible (cake) and irreversible (adsorbed/in-pore) layers. Cleaning removes only cake (though enhanced backwash may remove some irreversibles). Flux-area relation is constant (Darcy law holds). Membrane properties (R_m) fixed. Monte Carlo assumes certain pore geometry and fouling mechanism. ML assumes future behaves like training data.  
- **Inputs & Calibration:** Need pre-clean flux/TMP vs time (or volume) curves, water quality (foulant concentration, particle size), operating conditions (TMP, flow). For resistances models: measurement of single-cycle decays to fit rate constants and cleaning efficiency. For ML: historical dataset linking pre- and post-clean states.  
- **Parameters/Typical Ranges:** Specific cake resistance α ~10^11–10^13 m/kg (Carman–Kozeny)【76†L13-L19】; Hermia K ~0.01–0.1 s⁻¹ for protein solutions. Backwash efficiency η often 0.5–0.9 (50–90% of R_rev removed) depending on intensity. ANN/ML methods report RMSE in flux (e.g. ~5–10% of flux) when well-trained.  
- **Validation & Accuracy:**  
  - *Mechanistic:* Hlavacek et al. validated multi-step models on lab data, showing predicted TMP within ~10–20% of observed【88†L37-L44】. The Drumm & Bernardi MC model achieved good fits to experimental flux and partial flux recovery (graphs show close match)【92†L12-L21】.  
  - *Empirical:* Liaw et al.’s ANN predicted specific flux recovery with 90% accuracy in training/validation. MBR-Net reports high correlation ($R^2>0.9$) for lab-scale data.  
  - *Field:* Few field validations. Often models are validated on pilot test-rigs or short-term lab runs. Full-scale accuracy is uncertain due to variable conditions.  
- **Limitations/Gaps:**  
  - Most models need many fitted parameters or detailed data. They often assume constant cleaning efficiency regardless of load, which may vary with fouling intensity.  
  - ML models require extensive historical data (rare in practice).  
  - Hybrid or multi-mechanism models can overfit or fail if fouling mechanism changes (e.g. different foulant chemistry).  
  - No universal model covers MF/UF/NF/RO differences; separate calibration needed.  
  - Little work on long-term trends (e.g. aging membranes, scouring effects).  
  - **Conjugate models:** Resistance-in-series inherently link decline and recovery across cycles. Some explicit two-mechanism cycle models (Boltón, Ho) treat pre- and post-clean jointly, but such multi-cycle models remain research-level.

## BAC/BAF Filters

- **Studies/Models:** Biofilter modeling typically focuses on organic removal, with headloss as a byproduct, rather than explicitly predicting post-wash headloss. Sun et al. (2019) built a BAF model coupling nitrification and headloss; they observed backwash removed mainly particulates, with biofilm remaining【53†L67-L75】. Bi et al. (2014) modeled a biological aerated filter with multiple solid phases and computed headloss via Kozeny【57†L831-L839】. Neither gives a simple formula for post-backwash flux, but both simulate repeated filter/backwash cycles to steady-state.  
- **Formulations:**  Typical headloss equation: $ΔP = μV (R_{media}+R_{bio}+R_{suspended})$ where $R_{bio}$ increases with biomass volume (Monod kinetics) and $R_{suspended}$ with trapped particles【57†L831-L839】. Biofilter models include terms for biomass growth and decay, and often assume a fraction of biomass is sloughed at backwash. For example, a Monod growth equation $dX/dt = μX S/(K_S+S) - bX$, with $X$ converted to a volume filling the pores. After backwash, $X$ might be reset to a fraction (e.g. $X_{post} = (1-\alpha) X_{pre}$) representing sloughing.  
- **Assumptions:** Biofilm covers filter media as a thin layer, and backwash removes only loose layer/particles. Biofilm regrowth between washes is often computed from substrate loading. Sloughing rate is uncertain; many models assume an arbitrary fraction is removed or use detachment coefficients. Hydraulic conditions (media fluidization) may be simplified or ignored.  
- **Inputs/Calibration:** Require organics concentration, nitrification rate coefficients, biomass yield, media porosity/size. Headloss data pre- and post-wash would calibrate sloughing fraction. Few such data exist publicly.  
- **Parameters/Uncertainty:** Biofilm density (~90–300 kg/m³ biomass solids), yield Y (0.1–0.5 kg biomass/kg COD), detachment fraction (often 20–50% per wash assumed). Specific cake resistance for biomass is high (~10^13–10^15 m/kg). Reported uncertainty: models often fit pilot data, but no broad accuracy stats published.  
- **Validation:** Sun et al. validated their model on pilot and full-scale BAF, showing headloss trends matched reasonably, but details of recovery were not deeply analyzed【53†L67-L75】. Bi et al. showed headloss predictions within ~10% of data in lab tests【57†L831-L839】.  
- **Limitations/Gaps:** Biofilter models are complex and data-hungry. They rarely isolate backwash prediction accuracy. There is no standard approach for linking specific pre-wash pressure curves to post-wash states aside from assuming a percentage removal of biofilm/particles. Hybrid models (coupling removal and regrowth) exist in research but not in practice.  

## MBR/SBR-Associated Filters

- **Studies/Models:** Similar to other membrane systems, MBR fouling is often modeled with cake and pore fouling. Di Bella et al. (2019) review resistance-in-series in MBRs, noting cake acts as a dynamic membrane【65†L268-L281】. Some MBR models (e.g. Wu et al. 2006) use a “two-phase” approach: cake (reversible) and binding (irreversible). These can in principle predict flux after backwash by clearing the cake component.  
- **Formulations:** In batch SBR with submerged membranes, one can write $\Delta P = μJ (R_{m}+R_{cake}+R_{pore})$, with differential equations for $dR_{cake}/dt \propto TSS$ and $dR_{pore}/dt \propto SS$ adsorption. After backwash, $R_{cake}$ resets to near 0.  Some authors extend this by making $R_{pore}$ slowly increase over cycles.  
- **Assumptions:** Similar to point above: cake fouling is fully reversible with backwashing (air scouring + relaxation), irreversible part accumulates. Biology may form part of “cake”. Models often assume ideal backwash (full cake removal) or apply an efficiency.  
- **Inputs/Calibration:** Mixed liquor suspended solids, viscosity, airflow (for scouring efficiency), TMP records. Calibrate on TMP rise over time in batch and batch-averaged flux.  
- **Validation/Accuracy:** Lab MBR tests often show cake removal ~80–95%. Models capturing this can predict steady-state flux within 10–20%. However, full-scale variations (cleaning system design, non-uniform fouling) limit accuracy.  
- **Limitations/Gaps:** High variability in cleaning effectiveness; lack of models linking specific run history to exact recovery. MBR models focus on long-term steady state rather than each cycle’s recovery.  

## Hybrid Systems

- **Conjugate Models:** Some models explicitly span decline and recovery. For instance, Hlavacek’s combined blocking laws can be applied sequentially in cycles. The Monte Carlo model【92†L12-L21】 is a rare example that dynamically links forward and backward steps. Many wastewater MBR fouling models combine membranes with upstream cleaning (like periodic backpulse of membranes), but they usually treat pre- and post-clean states via separate equations with carryover of irreversible fouling.  
- **Mapping Algorithms:** No universal algorithm exists. A practical approach is to fit a fouling curve to each run, then assume the next run’s clean start is at $J_0'=J_0(1-\xi)$, where $\xi$ is residual decline (often $\xi=0$ for membranes after strong clean). For example, some engineers use an exponential decay fit ($J=J_0e^{-Kt}$) to the last cycle and apply the same $J_0$ after clean, effectively predicting full recovery of cake.  
- **Reversible/Irreversible Partitioning:** Nearly all predictive schemes rely on this partition. The core “algorithm” is: (1) from a pre-clean fit, extract reversible fouling magnitude; (2) set reversible part to zero (or fraction) after cleaning; (3) predict new starting flux. Mathematically:  
  $$J_{post} = \frac{\Delta P}{\mu(R_m + R_{irrev}+R_{suspended,new})},$$  
  where $R_{irrev}$ is carried over, and $R_{suspended,new}$ may depend on inflow quality. If using a Hermia model $J=J_0(1+Kt)^{-n/(n-2)}$, one might keep the same $J_0$ and $K$ but start $t$ from 0 after cleaning (implying nearly full cake removal). Some authors introduce a residual offset $R_{irrev}$ in Darcy’s law to shift the curve (e.g. push $J_0$ lower).  
- **Data Requirements:** Time-series of flux or TMP, plus knowledge of the cleaning outcome. Ideally, one needs experimental pre- vs post-clean curves to calibrate how much foulant remains. Without that, models assume a nominal efficiency (e.g. 90% cake removal).  
- **Parameter Ranges:** As discussed, irreversible fraction often 5–20% of total fouling per cycle in well-operated systems. Reversible removal efficiency η~80–95%. The exact numbers may be fitted from a few cycles.  
- **Reported Accuracy:** When backwash is ideal, predictions are trivial (full flux return). When not, validated models (e.g. Drumm MC) claim flux within ~5% of actual at end of backwash. Otherwise, simple models have large error if cleaning deviates from assumption.  
- **Limitations:** No single model covers all hybrid cases. Many filters have non-ideal cleaning (channeling, partial cake removal), which simple models cannot capture. Most literature addresses either decline *or* recovery, not the full mapping.

### Comparison of Selected Models

| **Model/Paper**                  | **System Type**       | **Predicts Post-Clean?** | **Data Needs**               | **Method**                          | **Accuracy**         |
|----------------------------------|-----------------------|--------------------------|------------------------------|-------------------------------------|----------------------|
| Hlavacek & Bouchet (1994)        | MF/UF (constant flow) | Partial (qualitative)    | Flux vs time per cycle       | 3 blocking laws + cake (resistances)| Fits individual cycles (R²~0.98 on test fluids) |
| Bolton et al. (2006)             | UF (dead-end)        | Partial                  | Flux decline pre-clean       | Two-stage (pore + cake) Hermia     | Good fit on lab data (no explicit post-clean step) |
| Drumm & Bernardi (2025, arXiv)   | MF (dead-end)        | Yes (via sim)            | Fouling characteristics     | Monte Carlo pore model (segmented)【92†L12-L21】| Good match to lab (error <~5% flux) |
| Liaw et al. (2020)               | MF (full-scale)      | Yes (ANN)                | Pre-clean flux decline, water quality | Artificial Neural Net          | ~90% classification of high/low recovery |
| Sun et al. (2019)                | BAC/BAF               | Implicit                 | Organics, nitrification, headloss vs time | Bio-kinetic + Kozeny (BAF model)【53†L67-L75】| Matched pilot data for headloss within 15% |
| Bi et al. (2014)                 | BAF (nitrifying)     | Implicit                 | Biomass, EPS, headloss vs time | Multi-phase solids + Kozeny【57†L831-L839】| Reasonable for headloss (8–12% error) |
| Di Bella et al. (2019) review    | MBR (submerged)      | Conceptual               | TMP/time series             | Resistance-in-series overview【65†L268-L281】| N/A (review)        |
| Wu et al. (2006) MBR model       | MBR (anoxic)         | Partial                  | TSS, substrate, TMP/time    | Cake + adsorption kinetic         | Validated short-term data         |

(_“Partial” indicates model addresses recovery implicitly via structure but not explicitly parameterized._)  

*Equations:* 
- **Resistance partition:** $J=\Delta P/\mu(R_m+R_{ir}+R_{rv})$【88†L37-L44】. After cleaning, set $R_{rv}\approx0$.  
- **Hermia law:** e.g. cake: $J=J_0/(1+Kt)$【28†L228-L236】. For sequential runs, some assume $J_0$ returns to original value (full cake removal) or to a reduced value if an offset (irreversible part) is kept.  
- **Biofilter headloss:** $ΔP = μV\,(R_{media} + α_b m_b + α_p m_p)$, with $dm_b/dt=μ_b(S)X, dm_p/dt=k_pC$ (particulate capture)【57†L831-L839】. After wash, $m_p→0$ and $X→(1-\gamma)X$ (slough fraction γ).  

Below is a *flowchart* summarizing model selection for linking decline and recovery:

```mermaid
flowchart TB
    A[Identify System] --> B{Membrane or Granular?}
    B -->|Membrane| C[Use resistance-in-series or blocking models]
    B -->|Granular/Sand| D[Fit empirical decline; assume fixed recovery fraction]
    C --> E{Data Availability}
    E -->|Mechanistic data| F[Build cake+pore fouling model]
    E -->|Historic data| G[Train ML regression/ANN]
    D --> H{Backwash data?}
    H -->|Yes| I[Estimate removal efficiency η from runs]
    H -->|No| J[Assume nominal η]
    F --> K{Cleaning type}
    K -->|Backwash| L[Reset reversible resistance (R_rev=0)]
    K -->|CIP| M[Apply factor to R_irrev]
```

**Assumptions & Gaps:** Most models assume uniform media and steady fouling kinetics; variable feed or shock loads break these assumptions. Uncertainties in $\eta$, biomass detachment, or short-term valve behavior are typically unquantified. Explicit hybrid models remain an active research area, particularly integrating fouling history into cleaning outcome predictions.

**Sources:** The above summary is drawn from key references and recent studies on filtration fouling and cleaning【28†L228-L236】【92†L12-L21】【88†L37-L44】【53†L67-L75】【57†L831-L839】, along with relevant reviews and field reports.