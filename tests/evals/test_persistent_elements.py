from navigation import (
    FindPersistentNavElements, 
    ConsolidateElements, 
    PersistedNavElementLM
)

DOM_STR = """
[1494]<div />
    [1496]<i />
    [1499]<p />
        Demo
    [1505]<span />
        Create account
[1512]<div />
    [1513]<span />
        [1517]<div />
            [1518]<p />
                Demo VgOcgPgdpi
            [1522]<i />
[1534]<a />
    Feed
[1544]<a />
    Snoozed
[1548]<a />
    Ignored
[1552]<a />
    Solved
[1560]<a />
    AutoFix
[1566]<hr />
[1571]<a />
    Repositories
[1583]<a />
    Containers
[1595]<a />
    Clouds
[1608]<a />
    Virtual Machines
[1619]<a />
    Domains & APIs
[1630]<a />
    Zen Firewall
[1640]<hr />
[1643]<a />
    Pentests
[1650]<a />
    Integrations
[1657]<a />
    Reports
[1675]<button />
[1687]<div />
    [1689]<p />
        [1691]<a />
            Settings
[1695]<i />
Repositories
[1703]<div />
    [1704]<div />
        [1705]<button />
[1719]<a />
    Docs
[1723]<a />
[1726]<div />
    [1727]<span />
        [1730]<div />
            [16]<img />
DV
Demo VgOcgPgdpi
1 active repo
[1766]<i />
1 member
|SCROLL|<div /> (horizontal 0%)
    [1778]<a />
        General
    [1780]<a />
        Users
    [1782]<a />
        Teams
    [1784]<a />
        Repositories
    [1786]<a />
        Clouds
    [1788]<a />
        Containers
    [1790]<a />
        Domains & APIs
    [1792]<a />
        Integrations
    [1794]<a />
        SLA
[1809]<div />
    [1810]<i />
    |SHADOW(open)|[14]<input type=text placeholder=Search />
[1816]<button />
    All Repositories
[1867]<button />
    Actions
[1882]<p />
    Actions
[1890]<button />
    Repository Settings
[1898]<button />
    Private Registry Connections
[1906]<button />
    Export Repositories
[1920]<button />
    Add repo
[1935]<p />
    [1939]<label />
        [15]<input type=checkbox value=false />
Repo name
[1953]<p />
    Domain
Sensitivity
Last scan
Activated
[1998]<a />
    [2003]<div />
    [18]<input type=checkbox />
    [2014]<span />
    demo-app-1
    Not public
    Sensitive
    Upgrade
[2176]<div role=button aria-label=Open Intercom Messenger />
    [2181]<svg /> <!-- SVG content collapsed -->
"""

if __name__ == "__main__":
    import asyncio
    from bupp.src.llm_models import openai_41
        
    async def run_find_persistent_nav_elements():
        """Single iteration of FindPersistentNavElements"""
        from navigation import FindPersistentNavElements
        
        res = await FindPersistentNavElements().ainvoke(
            model=openai_41(),
            prompt_args={
                "dom_str": DOM_STR
            },
            dry_run=False,
            clean_res=lambda s: s.replace("**","")
        )
        return res.persistent_el_list
    
    async def find_persisted_components(num_iterations=1) -> PersistedNavElementLM:
        """Run multiple iterations concurrently and consolidate results"""
        from navigation import ConsolidateElements
        
        # Run multiple iterations concurrently
        tasks = [asyncio.create_task(run_find_persistent_nav_elements()) for _ in range(num_iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if all results are equal
            # if len(results) > 1 and all(res == results[0] for res in results):
            #     print("All PersistentNavElement results are equal, skipping consolidation.")
            #     return results[0]
            
        # Create dictionary of ele_count: ele
        responses_dict = {}
        
        # Prepare LLM responses string for consolidation
        delimiter = "========================================"
        llm_responses = []
        ele_count = 1
        for i, res in enumerate(results, start=1):
            response_section = f"Response {i}:\n"
            for element in res:
                responses_dict[ele_count] = element
                response_section += f"{ele_count}. {element}\n"
                ele_count += 1
            llm_responses.append(response_section)
        
        llm_str = delimiter.join(llm_responses)
        print(llm_str)

        # Consolidate responses using LLM
        consolidated_res = await ConsolidateElements().ainvoke(
            model=openai_41(),
            prompt_args={
                "n": len(results),
                "dom_str": DOM_STR,
                "llm_responses": llm_str
            },
        )

        # Extract and return the final list of PersistentNavElement
        final_elements = []
        print(consolidated_res.response_indices)
        for index_str in consolidated_res.response_indices:
            index = int(index_str)
            if index in responses_dict:
                final_elements.extend(responses_dict[index])
        
        # Print the final consolidated results
        print("Final consolidated PersistentNavElement list:")
        for i, element in enumerate(final_elements, 1):
            print(f"{i}. {element}")
        
        return final_elements

    # Run the multi-threaded version for 3 iterations
    asyncio.run(find_persisted_components(3))