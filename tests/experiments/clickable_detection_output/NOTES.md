WIP to improve on BrowserUse DOM representation
Main line of attack is to try to reduce number of interactive elements. The problem is browser-use treats nested child elements as different clickables, when the click event is only triggered on the parent

https://claude.ai/chat/2b85a7bd-a1e6-459d-8c0d-12f8bbc95446

For example (https://www.ca.kayak.com/?ispredir=true)
[1184]<div role=group aria-label=Orlando />
            [1188]<div />
            [1191]<div />
                [1192]<h3 />
                    [1193]<a />
                        Orlando
                [1195]<div />
                    [1196]<div />
                        [1197]<div />
                            3h 7m, direct
                        [1199]<div />
                            [1200]<span />
                                Start date
                                Mon 26/1
                            <svg role=presentation /> <!-- SVG content collapsed -->
                            [1206]<span />
                                End date
                                Fri 30/1
                    [1210]<div />
                        [1211]<div />
                            [1212]<div />

Only the Top-level parent is clickable here
Tmrw: 
-> do this for every new page
-> dont on update, update the DOM and this representation so we keep it in-sync
[NAV]
├── [MENU_TOGGLE] Open main navigation
├── [HOME_LINK] Go to the kayak homepage
├── [MENU_TOGGLE] Open Trips drawer
└── [ACTION] Sign in

[SEARCH_FORM] flight
├── [TAB_GROUP]
│   ├── [TAB] Flights (active)
│   ├── [TAB] Stays
│   ├── [TAB] Cars
│   ├── [TAB] Flight+Hotel
│   └── [TAB] AI Mode (beta)
├── [FORM_CONTROLS]
│   ├── [DROPDOWN] Trip type: Round-trip
│   ├── [DROPDOWN] Bags: 0
│   ├── [INPUT] Origin: Toronto (YTO)
│   ├── [SWAP_BUTTON] Swap origin/destination
│   ├── [INPUT] Destination: (empty)
│   ├── [DATE_PICKER] Departure
│   ├── [DATE_PICKER] Return
│   ├── [DROPDOWN] Travelers: 1 adult, Economy
│   └── [SUBMIT] Search

[PROMO_BANNER]
├── [STAT] 41,000,000+ searches this week
└── [STAT] 1M+ ratings on our app

[DEALS_SECTION] Travel deals under C$ 298
├── [LINK] Explore more
├── [DEAL_CARD] Halifax
│   ├── 2h 40m, direct
│   ├── Thu 15/1 → Mon 19/1
│   └── from C$ 118
├── [DEAL_CARD] Fort Lauderdale
│   ├── 3h 25m, direct
│   ├── Sun 18/1 → Thu 22/1
│   └── from C$ 139
├── [DEAL_CARD] Orlando
│   ├── 3h 7m, direct
│   ├── Fri 16/1 → Fri 23/1
│   └── from C$ 182
└── [DEAL_CARD] Fort Myers
    ├── 3h 25m, direct
    ├── Wed 21/1 → Sun 25/1
    └── from C$ 210

[FEATURES_SECTION] For travel pros
├── [FEATURE_CARD] KAYAK.ai (BETA)
│   └── Get travel questions answered
├── [FEATURE_CARD] Best Time to Travel
│   └── Know when to save
├── [FEATURE_CARD] Explore
│   └── See destinations on your budget
└── [FEATURE_CARD] Trips
    └── Keep all your plans in one place