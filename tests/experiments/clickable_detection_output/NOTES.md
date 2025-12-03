WIP to improve on BrowserUse DOM representation
Main line of attack is to try to reduce number of interactive elements. The problem is browser-use treats nested child elements as different clickables, when the click event is only triggered on the parent

https://claude.ai/chat/2b85a7bd-a1e6-459d-8c0d-12f8bbc95446

For example (https://www.ca.kayak.com/?ispredir=true):
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
