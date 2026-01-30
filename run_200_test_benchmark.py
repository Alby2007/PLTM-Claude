"""
Test generator for 200-test comprehensive benchmark
Generates all 200 test cases programmatically
"""

import asyncio
import time
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.reconciliation.conflict_detector import ConflictDetector


class ComprehensiveBenchmark:
    """200-test comprehensive benchmark suite"""
    
    def __init__(self):
        self.store = SQLiteGraphStore(":memory:")
        self.extractor = RuleBasedExtractor()
        self.detector = ConflictDetector(self.store)
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": 0,
            "failures": [],
            "categories": {}
        }
    
    async def run_test(self, test_num: int, name: str, category: str, test_func):
        """Run a single test"""
        try:
            await self.store.connect()
            await test_func()
            self.results["passed"] += 1
            if category not in self.results["categories"]:
                self.results["categories"][category] = {"passed": 0, "failed": 0}
            self.results["categories"][category]["passed"] += 1
            print(f"✓ Test {test_num:03d}: {name}")
            return True
        except AssertionError as e:
            self.results["failed"] += 1
            if category not in self.results["categories"]:
                self.results["categories"][category] = {"passed": 0, "failed": 0}
            self.results["categories"][category]["failed"] += 1
            self.results["failures"].append({
                "test": test_num,
                "name": name,
                "category": category,
                "error": str(e)
            })
            print(f"✗ Test {test_num:03d}: {name} - FAILED")
            return False
        except Exception as e:
            self.results["errors"] += 1
            if category not in self.results["categories"]:
                self.results["categories"][category] = {"passed": 0, "failed": 0}
            self.results["categories"][category]["failed"] += 1
            self.results["failures"].append({
                "test": test_num,
                "name": name,
                "category": category,
                "error": f"ERROR: {str(e)}"
            })
            print(f"✗ Test {test_num:03d}: {name} - ERROR: {e}")
            return False
        finally:
            self.results["total"] += 1
            try:
                await self.store.close()
            except:
                pass
    
    async def test_opposite_predicates(self, stmt1: str, stmt2: str):
        """Generic opposite predicate test"""
        atoms1 = self.extractor.extract(stmt1, "user_1")
        if atoms1:
            await self.store.insert_atom(atoms1[0])
        
        atoms2 = self.extractor.extract(stmt2, "user_1")
        if atoms2:
            conflicts = await self.detector.find_conflicts(atoms2[0])
            assert len(conflicts) > 0, f"Should detect conflict: {stmt1} vs {stmt2}"
    
    async def test_exclusive_predicates(self, stmt1: str, stmt2: str):
        """Generic exclusive predicate test"""
        atoms1 = self.extractor.extract(stmt1, "user_1")
        if atoms1:
            await self.store.insert_atom(atoms1[0])
        
        atoms2 = self.extractor.extract(stmt2, "user_1")
        if atoms2:
            conflicts = await self.detector.find_conflicts(atoms2[0])
            assert len(conflicts) > 0, f"Should detect exclusive conflict: {stmt1} vs {stmt2}"
    
    async def test_no_conflict(self, stmt1: str, stmt2: str):
        """Generic no-conflict test"""
        atoms1 = self.extractor.extract(stmt1, "user_1")
        if atoms1:
            await self.store.insert_atom(atoms1[0])
        
        atoms2 = self.extractor.extract(stmt2, "user_1")
        if atoms2:
            conflicts = await self.detector.find_conflicts(atoms2[0])
            assert len(conflicts) == 0, f"Should NOT conflict: {stmt1} vs {stmt2}"
    
    def generate_all_tests(self):
        """Generate all 200 test cases"""
        tests = []
        test_num = 1
        
        # Category 1: Opposite Predicates (30 tests)
        opposite_pairs = [
            ("I like Python", "I dislike Python"),
            ("I love JavaScript", "I hate JavaScript"),
            ("I enjoy coding", "I dislike coding"),
            ("I prefer TypeScript", "I avoid TypeScript"),
            ("I want to learn Rust", "I don't want to learn Rust"),
            ("I support remote work", "I oppose remote work"),
            ("I agree with TDD", "I disagree with TDD"),
            ("I trust AI systems", "I distrust AI systems"),
            ("I respect functional programming", "I disrespect functional programming"),
            ("I admire clean code", "I despise clean code"),
            ("I like React", "I dislike React"),
            ("I love Vue", "I hate Vue"),
            ("I enjoy Angular", "I dislike Angular"),
            ("I prefer Svelte", "I avoid Svelte"),
            ("I like Docker", "I dislike Docker"),
            ("I love Kubernetes", "I hate Kubernetes"),
            ("I enjoy DevOps", "I dislike DevOps"),
            ("I prefer microservices", "I avoid microservices"),
            ("I like GraphQL", "I dislike GraphQL"),
            ("I love REST APIs", "I hate REST APIs"),
            ("I enjoy testing", "I dislike testing"),
            ("I prefer TDD", "I avoid TDD"),
            ("I like pair programming", "I dislike pair programming"),
            ("I love code reviews", "I hate code reviews"),
            ("I enjoy refactoring", "I dislike refactoring"),
            ("I prefer agile", "I avoid agile"),
            ("I like scrum", "I dislike scrum"),
            ("I love kanban", "I hate kanban"),
            ("I enjoy CI/CD", "I dislike CI/CD"),
            ("I prefer automation", "I avoid automation"),
        ]
        
        for stmt1, stmt2 in opposite_pairs:
            tests.append((
                test_num,
                f"Opposite: {stmt1} vs {stmt2}",
                "Opposite Predicates",
                lambda s1=stmt1, s2=stmt2: self.test_opposite_predicates(s1, s2)
            ))
            test_num += 1
        
        # Category 2: Exclusive Predicates (40 tests)
        exclusive_pairs = [
            ("I live in Seattle", "I live in San Francisco"),
            ("I work at Google", "I work at Meta"),
            ("I work at Amazon", "I work at Microsoft"),
            ("I live in New York", "I live in London"),
            ("I work at Apple", "I work at Tesla"),
            ("I live in Tokyo", "I live in Paris"),
            ("I work at Netflix", "I work at Spotify"),
            ("I live in Berlin", "I live in Amsterdam"),
            ("I work at Uber", "I work at Lyft"),
            ("I live in Austin", "I live in Denver"),
            ("I work at Airbnb", "I work at Booking"),
            ("I live in Singapore", "I live in Hong Kong"),
            ("I work at Stripe", "I work at Square"),
            ("I live in Sydney", "I live in Melbourne"),
            ("I work at Salesforce", "I work at Oracle"),
            ("I live in Toronto", "I live in Vancouver"),
            ("I work at IBM", "I work at Intel"),
            ("I live in Boston", "I live in Chicago"),
            ("I work at Adobe", "I work at Autodesk"),
            ("I live in Miami", "I live in Atlanta"),
            ("I work at Shopify", "I work at Etsy"),
            ("I live in Dublin", "I live in Edinburgh"),
            ("I work at Zoom", "I work at Slack"),
            ("I live in Barcelona", "I live in Madrid"),
            ("I work at Twitter", "I work at LinkedIn"),
            ("I live in Stockholm", "I live in Copenhagen"),
            ("I work at GitHub", "I work at GitLab"),
            ("I live in Zurich", "I live in Geneva"),
            ("I work at Dropbox", "I work at Box"),
            ("I live in Seoul", "I live in Beijing"),
            ("I work at PayPal", "I work at Venmo"),
            ("I live in Mumbai", "I live in Delhi"),
            ("I work at Snap", "I work at Pinterest"),
            ("I live in São Paulo", "I live in Rio"),
            ("I work at Reddit", "I work at Discord"),
            ("I live in Moscow", "I live in St Petersburg"),
            ("I work at Twitch", "I work at YouTube"),
            ("I live in Cairo", "I live in Dubai"),
            ("I work at Figma", "I work at Canva"),
            ("I live in Mexico City", "I live in Buenos Aires"),
        ]
        
        for stmt1, stmt2 in exclusive_pairs:
            tests.append((
                test_num,
                f"Exclusive: {stmt1} vs {stmt2}",
                "Exclusive Predicates",
                lambda s1=stmt1, s2=stmt2: self.test_exclusive_predicates(s1, s2)
            ))
            test_num += 1
        
        # Category 3: Contextual No-Conflicts (30 tests)
        contextual_pairs = [
            ("I like Python for data science", "I like JavaScript for web dev"),
            ("I prefer coffee in the morning", "I prefer tea in the evening"),
            ("I like running outdoors", "I like swimming indoors"),
            ("I enjoy reading fiction", "I enjoy reading non-fiction"),
            ("I prefer Mac for development", "I prefer Windows for gaming"),
            ("I like Italian food for dinner", "I like Japanese food for lunch"),
            ("I enjoy jazz when relaxing", "I enjoy rock when exercising"),
            ("I prefer async code for I/O", "I prefer sync code for CPU tasks"),
            ("I like SQL for analytics", "I like NoSQL for caching"),
            ("I enjoy hiking in summer", "I enjoy skiing in winter"),
            ("I prefer email for formal communication", "I prefer Slack for quick chats"),
            ("I like REST for simple APIs", "I like GraphQL for complex queries"),
            ("I enjoy podcasts while commuting", "I enjoy audiobooks while traveling"),
            ("I prefer dark mode at night", "I prefer light mode during day"),
            ("I like Python for scripting", "I like Go for services"),
            ("I enjoy chess for strategy", "I enjoy poker for probability"),
            ("I prefer vim for editing", "I prefer IDE for debugging"),
            ("I like Redis for caching", "I like Postgres for persistence"),
            ("I enjoy biking for exercise", "I enjoy yoga for flexibility"),
            ("I prefer Docker for dev", "I prefer Kubernetes for prod"),
            ("I like unit tests for functions", "I like integration tests for APIs"),
            ("I enjoy Twitter for news", "I enjoy YouTube for tutorials"),
            ("I prefer functional for data", "I prefer OOP for systems"),
            ("I like Markdown for docs", "I like LaTeX for papers"),
            ("I enjoy standup comedy for laughs", "I enjoy documentaries for learning"),
            ("I prefer Git for version control", "I prefer Jira for project management"),
            ("I like Prometheus for metrics", "I like Grafana for visualization"),
            ("I enjoy breakfast at 7am", "I enjoy dinner at 7pm"),
            ("I prefer meetings in morning", "I prefer deep work in afternoon"),
            ("I like Python 3.11 for speed", "I like Python 3.12 for features"),
        ]
        
        for stmt1, stmt2 in contextual_pairs:
            tests.append((
                test_num,
                f"Context: {stmt1} + {stmt2}",
                "Contextual No-Conflicts",
                lambda s1=stmt1, s2=stmt2: self.test_no_conflict(s1, s2)
            ))
            test_num += 1
        
        # Category 4: Temporal & Refinements (30 tests)
        temporal_pairs = [
            ("I used to like Java", "I like Python"),
            ("I previously worked at Google", "I work at Meta"),
            ("I formerly lived in Seattle", "I live in San Francisco"),
            ("I used to prefer OOP", "I prefer functional programming"),
            ("I previously used vim", "I use VS Code"),
            ("I formerly liked Windows", "I like Mac"),
            ("I used to enjoy PHP", "I enjoy TypeScript"),
            ("I previously preferred SQL", "I prefer NoSQL"),
            ("I formerly used SVN", "I use Git"),
            ("I used to like waterfall", "I like agile"),
            ("I like programming", "I like Python programming"),
            ("I enjoy music", "I enjoy jazz music"),
            ("I prefer databases", "I prefer SQL databases"),
            ("I like sports", "I like basketball"),
            ("I enjoy reading", "I enjoy reading sci-fi"),
            ("I prefer frameworks", "I prefer React frameworks"),
            ("I like languages", "I like programming languages"),
            ("I enjoy games", "I enjoy video games"),
            ("I prefer tools", "I prefer development tools"),
            ("I like food", "I like Italian food"),
            ("I will learn Rust next year", "I like Python"),
            ("I plan to move to Seattle", "I live in San Francisco"),
            ("I might try Go", "I like Python"),
            ("I could learn Haskell", "I like JavaScript"),
            ("I may switch to Mac", "I use Windows"),
            ("I will start running", "I enjoy swimming"),
            ("I plan to learn guitar", "I play piano"),
            ("I might try meditation", "I do yoga"),
            ("I could learn Spanish", "I speak English"),
            ("I may start blogging", "I write code"),
        ]
        
        for stmt1, stmt2 in temporal_pairs:
            tests.append((
                test_num,
                f"Temporal: {stmt1} + {stmt2}",
                "Temporal & Refinements",
                lambda s1=stmt1, s2=stmt2: self.test_no_conflict(s1, s2)
            ))
            test_num += 1
        
        # Category 5: Duplicates & Similar (30 tests)
        duplicate_pairs = [
            ("I like Python", "I like Python"),
            ("I work at Google", "I work at Google"),
            ("I live in Seattle", "I live in Seattle"),
            ("I enjoy coding", "I enjoy coding"),
            ("I prefer TypeScript", "I prefer TypeScript"),
            ("I love JavaScript", "I love JavaScript"),
            ("I hate bugs", "I hate bugs"),
            ("I use VS Code", "I use VS Code"),
            ("I prefer Mac", "I prefer Mac"),
            ("I enjoy coffee", "I enjoy coffee"),
            ("I like Python", "I really like Python"),
            ("I work at Google", "I currently work at Google"),
            ("I live in Seattle", "I am living in Seattle"),
            ("I enjoy coding", "I truly enjoy coding"),
            ("I prefer TypeScript", "I strongly prefer TypeScript"),
            ("I love JavaScript", "I absolutely love JavaScript"),
            ("I hate bugs", "I really hate bugs"),
            ("I use VS Code", "I always use VS Code"),
            ("I prefer Mac", "I definitely prefer Mac"),
            ("I enjoy coffee", "I really enjoy coffee"),
            ("I like Python", "I like python"),
            ("I work at Google", "I work at google"),
            ("I live in Seattle", "I live in seattle"),
            ("I enjoy coding", "I enjoy Coding"),
            ("I prefer TypeScript", "I prefer typescript"),
            ("I love JavaScript", "I love javascript"),
            ("I use VS Code", "I use vs code"),
            ("I prefer Mac", "I prefer mac"),
            ("I enjoy coffee", "I enjoy Coffee"),
            ("I like React", "I like react"),
        ]
        
        for stmt1, stmt2 in duplicate_pairs:
            tests.append((
                test_num,
                f"Duplicate: {stmt1} = {stmt2}",
                "Duplicates & Similar",
                lambda s1=stmt1, s2=stmt2: self.test_no_conflict(s1, s2)
            ))
            test_num += 1
        
        # Category 6: Edge Cases (20 tests)
        edge_pairs = [
            ("I like C++", "I dislike C++"),
            ("I like C#", "I dislike C#"),
            ("I like F#", "I dislike F#"),
            ("I like .NET", "I dislike .NET"),
            ("I like Node.js", "I dislike Node.js"),
            ("I like Vue.js", "I dislike Vue.js"),
            ("I like Next.js", "I dislike Next.js"),
            ("I like D3.js", "I dislike D3.js"),
            ("I like Three.js", "I dislike Three.js"),
            ("I like p5.js", "I dislike p5.js"),
            ("I like @angular/core", "I dislike @angular/core"),
            ("I like react-router", "I dislike react-router"),
            ("I like lodash/fp", "I dislike lodash/fp"),
            ("I like express.js", "I dislike express.js"),
            ("I like socket.io", "I dislike socket.io"),
            ("I like styled-components", "I dislike styled-components"),
            ("I like redux-saga", "I dislike redux-saga"),
            ("I like webpack.config.js", "I dislike webpack.config.js"),
            ("I like babel.config.js", "I dislike babel.config.js"),
            ("I like jest.config.js", "I dislike jest.config.js"),
        ]
        
        for stmt1, stmt2 in edge_pairs:
            tests.append((
                test_num,
                f"Edge: {stmt1} vs {stmt2}",
                "Edge Cases",
                lambda s1=stmt1, s2=stmt2: self.test_opposite_predicates(s1, s2)
            ))
            test_num += 1
        
        # Category 7: Multi-Step (10 tests)
        multi_step_tests = [
            ("I like Python", "I love Python", False),
            ("I work at Google", "I work at Google in Seattle", False),
            ("I prefer TypeScript", "I strongly prefer TypeScript", False),
            ("I enjoy coding", "I really enjoy coding in Python", False),
            ("I live in Seattle", "I live in Seattle, Washington", False),
            ("I use VS Code", "I use VS Code for Python", False),
            ("I like coffee", "I like coffee in the morning", False),
            ("I enjoy reading", "I enjoy reading technical books", False),
            ("I prefer Mac", "I prefer Mac for development", False),
            ("I like React", "I like React for web development", False),
        ]
        
        for stmt1, stmt2, should_conflict in multi_step_tests:
            tests.append((
                test_num,
                f"Multi-step: {stmt1} → {stmt2}",
                "Multi-Step",
                lambda s1=stmt1, s2=stmt2: self.test_no_conflict(s1, s2)
            ))
            test_num += 1
        
        # Category 8: Real-World (10 tests)
        real_world_pairs = [
            ("I am allergic to peanuts", "I am allergic to shellfish"),
            ("I take medication for diabetes", "I take medication for hypertension"),
            ("I prefer email communication", "I prefer phone calls"),
            ("I am vegetarian", "I eat chicken"),
            ("I have a PhD in Computer Science", "I have a Master's in Mathematics"),
            ("I speak English", "I speak Spanish"),
            ("I am left-handed", "I am right-handed"),
            ("I have 5 years of Python experience", "I have 3 years of JavaScript experience"),
            ("I graduated from MIT", "I graduated from Stanford"),
            ("I am a morning person", "I am a night owl"),
        ]
        
        for stmt1, stmt2 in real_world_pairs:
            # First 4 should not conflict (multiple allergies, medications, etc.)
            # Last 6 should conflict (exclusive attributes)
            if test_num <= 194:
                tests.append((
                    test_num,
                    f"Real-world: {stmt1} + {stmt2}",
                    "Real-World",
                    lambda s1=stmt1, s2=stmt2: self.test_no_conflict(s1, s2)
                ))
            else:
                tests.append((
                    test_num,
                    f"Real-world: {stmt1} vs {stmt2}",
                    "Real-World",
                    lambda s1=stmt1, s2=stmt2: self.test_exclusive_predicates(s1, s2)
                ))
            test_num += 1
        
        return tests
    
    async def run_all(self):
        """Run all 200 tests"""
        print("\n" + "="*70)
        print("RUNNING 200-TEST COMPREHENSIVE BENCHMARK")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        # Generate all tests
        tests = self.generate_all_tests()
        
        print(f"Generated {len(tests)} tests\n")
        
        # Run tests
        for test_num, name, category, test_func in tests:
            self.store = SQLiteGraphStore(":memory:")
            self.detector = ConflictDetector(self.store)
            await self.run_test(test_num, name, category, test_func)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(f"\nTotal Tests:    {self.results['total']}")
        print(f"Passed:         {self.results['passed']} ✓")
        print(f"Failed:         {self.results['failed']} ✗")
        print(f"Errors:         {self.results['errors']} ⚠")
        print(f"\nAccuracy:       {self.results['passed']}/{self.results['total']} ({self.results['passed']/self.results['total']*100:.1f}%)")
        print(f"Duration:       {duration:.2f} seconds")
        print(f"Avg per test:   {duration/self.results['total']*1000:.1f} ms")
        
        # Category breakdown
        print("\n" + "="*70)
        print("CATEGORY BREAKDOWN")
        print("="*70)
        for category, stats in sorted(self.results["categories"].items()):
            total = stats["passed"] + stats["failed"]
            accuracy = stats["passed"] / total * 100 if total > 0 else 0
            print(f"{category:25s}: {stats['passed']:3d}/{total:3d} ({accuracy:5.1f}%)")
        
        if self.results['failures']:
            print("\n" + "="*70)
            print("FAILURES")
            print("="*70)
            for failure in self.results['failures'][:10]:  # Show first 10
                print(f"\nTest {failure['test']:03d} [{failure['category']}]: {failure['name']}")
                print(f"  Error: {failure['error']}")
            if len(self.results['failures']) > 10:
                print(f"\n... and {len(self.results['failures']) - 10} more failures")
        
        print("\n" + "="*70)
        
        # Return success if >95% pass rate
        pass_rate = self.results['passed'] / self.results['total']
        if pass_rate >= 0.95:
            print(f"✓ BENCHMARK PASSED (>95% accuracy)")
        else:
            print(f"✗ BENCHMARK FAILED (<95% accuracy)")
        print("="*70 + "\n")
        
        return pass_rate >= 0.95


async def main():
    """Main entry point"""
    runner = ComprehensiveBenchmark()
    success = await runner.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
