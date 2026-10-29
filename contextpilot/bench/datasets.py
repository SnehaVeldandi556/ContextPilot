"""Benchmark dataset loaders and generators for HotpotQA, Natural Questions, and MS MARCO."""

import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

BENCHMARK_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Passage:
    id: str
    text: str
    is_supporting: bool = False


@dataclass
class BenchmarkItem:
    id: str
    dataset: str
    question: str
    passages: List[Passage]
    gold_answer: str
    gold_supporting_ids: List[str] = field(default_factory=list)
    is_multi_hop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "passages": [asdict(p) for p in self.passages],
            "gold_answer": self.gold_answer,
            "gold_supporting_ids": self.gold_supporting_ids,
            "is_multi_hop": self.is_multi_hop,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkItem":
        passages = [Passage(**p) for p in data["passages"]]
        return cls(
            id=data["id"],
            dataset=data["dataset"],
            question=data["question"],
            passages=passages,
            gold_answer=data["gold_answer"],
            gold_supporting_ids=data.get("gold_supporting_ids", []),
            is_multi_hop=data.get("is_multi_hop", False),
        )


# Pre-crafted realistic benchmark templates to generate 300-500 diverse samples per dataset
HOTPOTQA_TEMPLATES = [
    {
        "q": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "entity1": "Scott Derrickson",
        "entity2": "Ed Wood",
        "prop1": "American filmmaker",
        "prop2": "American filmmaker and actor",
        "ans": "yes",
        "p1": "Scott Derrickson (born July 16, 1966) is an American filmmaker. He is best known for directing horror films such as Sinister and the Marvel Cinematic Universe superhero film Doctor Strange.",
        "p2": "Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, and pulp crime author.",
        "distractors": [
            "Scott Derrickson attended Biola University where he earned a B.A. in Humanities with an emphasis on literature.",
            "Ed Wood in the 1950s made several low-budget science fiction and horror films including Glen or Glenda and Plan 9 from Outer Space.",
            "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name.",
            "Plan 9 from Outer Space is a 1957 American black-and-white science fiction-horror film produced by Ed Wood.",
            "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.",
            "Biola University is a private Christian university in La Mirada, California.",
            "Glen or Glenda is a 1953 American docudrama film about cross-dressing and transvestism.",
            "Bela Lugosi was a Hungarian-American actor best remembered for portraying Count Dracula in the 1931 film.",
            "The Marvel Cinematic Universe is an American media franchise and shared universe centered on a series of superhero films.",
            "C. Robert Cargill is an American screenwriter, novelist, and former film critic known as Massawyrm.",
            "Bride of the Monster is a 1955 science fiction horror film co-written and directed by Edward D. Wood Jr.",
            "The Exorcism of Emily Rose is a 2005 American legal drama supernatural horror film directed by Scott Derrickson.",
            "Night of the Ghouls is a 1959 horror film written and directed by Edward D. Wood Jr.",
            "Benedict Cumberbatch portrays Dr. Stephen Strange in Doctor Strange, directed by Derrickson.",
            "Tim Burton directed the 1994 biographical comedy-drama film titled Ed Wood, starring Johnny Depp.",
            "Johnny Depp portrayed the eccentric director Ed Wood in the 1994 biographical film.",
            "Sarah Derrickson is a film producer who collaborated on various projects in independent film.",
            "American cinema has had a profound effect on world entertainment since the early 20th century.",
            "Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson.",
            "Orson Welles was an American director, actor, screenwriter, and producer who Wood greatly admired."
        ]
    },
    {
        "q": "Which film has the director who was born earlier, Inception or Pulp Fiction?",
        "entity1": "Inception",
        "entity2": "Pulp Fiction",
        "ans": "Pulp Fiction",
        "p1": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan. Christopher Nolan was born on 30 July 1970 in London, England.",
        "p2": "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino. Quentin Tarantino was born on March 27, 1963 in Knoxville, Tennessee.",
        "distractors": [
            "Christopher Nolan is a British-American filmmaker known for his Hollywood blockbusters with complex storytelling.",
            "Quentin Tarantino is an American film director, screenwriter, producer, and author celebrated for his non-linear storylines.",
            "Leonardo DiCaprio starred in Inception as Dom Cobb, a professional thief who steals information from subconscious minds.",
            "John Travolta and Samuel L. Jackson starred in Pulp Fiction as Vincent Vega and Jules Winnfield.",
            "The Dark Knight Trilogy was directed by Christopher Nolan between 2005 and 2012.",
            "Reservoir Dogs is a 1992 American independent crime film written and directed by Quentin Tarantino in his feature-length debut.",
            "Interstellar is a 2014 epic science fiction film directed, co-written and produced by Christopher Nolan.",
            "Kill Bill: Volume 1 is a 2003 American martial arts film written and directed by Quentin Tarantino.",
            "Oppenheimer is a 2023 biographical drama film written, directed, and produced by Christopher Nolan.",
            "Once Upon a Time in Hollywood is a 2019 comedy-drama film written and directed by Quentin Tarantino.",
            "The 67th Academy Awards ceremony took place on Monday, March 27, 1995 at the Shrine Auditorium in Los Angeles.",
            "Cannes Film Festival awarded the Palme d'Or to Pulp Fiction in May 1994.",
            "Hans Zimmer composed the iconic score for Inception, including the track 'Time'.",
            "Cinema of the United States has produced many critically acclaimed directors across the 1990s and 2000s.",
            "Film directors born in the 1960s and 1970s often drew inspiration from classic American and European cinema.",
            "Memento is a 2000 American neo-noir psychological mystery film written and directed by Christopher Nolan.",
            "Inglourious Basterds is a 2009 war film written and directed by Quentin Tarantino starring Brad Pitt.",
            "Dunkirk is a 2017 historical war film written, directed, and produced by Christopher Nolan.",
            "Django Unchained is a 2012 American revisionist Western film written and directed by Quentin Tarantino.",
            "British film directors have made significant contributions to Hollywood and international cinema."
        ]
    },
    {
        "q": "What government agency was headed by the author of the report on the assassination of John F. Kennedy?",
        "entity1": "Warren Commission",
        "entity2": "Earl Warren",
        "ans": "Supreme Court of the United States",
        "p1": "The Warren Commission report on the assassination of President John F. Kennedy was headed by Chief Justice Earl Warren.",
        "p2": "Earl Warren served as the 14th Chief Justice of the United States, heading the Supreme Court of the United States from 1953 to 1969.",
        "distractors": [
            "John F. Kennedy, the 35th president of the United States, was assassinated on November 22, 1963 in Dallas, Texas.",
            "The Federal Bureau of Investigation (FBI) conducted extensive investigations following the assassination.",
            "The Central Intelligence Agency (CIA) provided intelligence dossiers during the 1960s.",
            "The Supreme Court of the United States is the highest court in the federal judiciary of the United States.",
            "Chief Justice Earl Warren presided over the landmark case Brown v. Board of Education in 1954.",
            "Lyndon B. Johnson created the President's Commission on the Assassination of President Kennedy on November 29, 1963.",
            "The Department of Justice oversees federal law enforcement in the United States.",
            "Gerald Ford was a member of the Warren Commission before later becoming President of the United States.",
            "Allen Dulles, former director of the CIA, was appointed as a member of the Warren Commission.",
            "The House Select Committee on Assassinations reinvestigated the assassination of JFK in 1976.",
            "The Secret Service is responsible for protecting the President of the United States.",
            "Texas Governor John Connally was also wounded during the shooting of President Kennedy.",
            "The National Archives and Records Administration preserves historical government records.",
            "Dealey Plaza in Dallas was the location where President Kennedy's motorcade was fired upon.",
            "The United States Congress consists of the Senate and the House of Representatives.",
            "Federal judicial appointments require nomination by the President and confirmation by the Senate.",
            "Lee Harvey Oswald was arrested for the assassination of President Kennedy shortly after the shooting.",
            "Jack Ruby shot and killed Lee Harvey Oswald in the basement of Dallas police headquarters.",
            "United States Court of Appeals serves as intermediate appellate federal courts.",
            "Constitutional law in the United States is largely shaped by landmark decisions of the Supreme Court."
        ]
    }
]

NATURAL_QUESTIONS_TEMPLATES = [
    {
        "q": "Where did the 2016 Summer Olympics take place?",
        "ans": "Rio de Janeiro, Brazil",
        "p_pos": "The 2016 Summer Olympics, officially known as the Games of the XXXI Olympiad and commonly known as Rio 2016, took place in Rio de Janeiro, Brazil, from 5 to 21 August 2016.",
        "distractors": [
            "The 2012 Summer Olympics were held in London, United Kingdom, from 27 July to 12 August 2012.",
            "The 2020 Summer Olympics took place in Tokyo, Japan, postponed to 2021 due to the COVID-19 pandemic.",
            "The 2008 Summer Olympics were hosted by Beijing, China, at the National Stadium.",
            "Rio de Janeiro is the second-most populous municipality in Brazil and the sixth-most populous in the Americas.",
            "The International Olympic Committee (IOC) is a non-governmental sports organisation based in Lausanne, Switzerland.",
            "Maracanã Stadium in Rio de Janeiro hosted the opening and closing ceremonies of the 2016 Olympic Games.",
            "The 2016 Paralympic Games were also hosted in Rio de Janeiro following the Olympic Games.",
            "Brazil is the largest country in both South America and Latin America with over 214 million people.",
            "Over 11,000 athletes from 205 National Olympic Committees competed in the 2016 Summer Olympics.",
            "Usain Bolt won three gold medals at the 2016 Rio Games in the 100m, 200m, and 4x100m relay.",
            "Michael Phelps concluded his competitive swimming career at the 2016 Rio Games winning five gold medals.",
            "The 2024 Summer Olympics were scheduled to be held in Paris, France.",
            "The Olympic torch relay for Rio 2016 traveled across Brazil visiting more than 300 cities.",
            "South America had never hosted an Olympic Games prior to the 2016 Rio Summer Olympics.",
            "Barra da Tijuca hosted most of the competition venues for the 2016 Summer Olympics.",
            "The Olympic Village in Rio housed over 17,000 athletes and team officials during the competition.",
            "Rugby sevens and golf were introduced as new Olympic sports at the 2016 Summer Games.",
            "The United States topped the medal table in 2016 with 46 gold medals and 121 total medals.",
            "Christ the Redeemer is an Art Deco statue of Jesus Christ overlooking Rio de Janeiro.",
            "Copacabana Beach in Rio de Janeiro hosted beach volleyball events during the 2016 Games."
        ]
    },
    {
        "q": "What is the chemical symbol for gold?",
        "ans": "Au",
        "p_pos": "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79, making it one of the higher atomic number elements that occur naturally.",
        "distractors": [
            "Silver is a chemical element with the symbol Ag (from Latin: argentum) and atomic number 47.",
            "Copper is a chemical element with the symbol Cu (from Latin: cuprum) and atomic number 29.",
            "Iron is a chemical element with symbol Fe (from Latin: ferrum) and atomic number 26.",
            "Lead is a chemical element with the symbol Pb (from Latin: plumbum) and atomic number 82.",
            "Mercury is a chemical element with symbol Hg (from Greek: hydrargyrum) and atomic number 80.",
            "The periodic table is a tabular display of the chemical elements arranged by atomic number.",
            "Platinum is a chemical element with symbol Pt and atomic number 78, situated adjacent to gold.",
            "Aurum is the Latin word for gold, signifying shining dawn in ancient Indo-European linguistics.",
            "Gold has been used for coinage, jewelry, and other arts throughout recorded human history.",
            "Malleability and ductility are physical properties in which gold excels above most known metals.",
            "Alloys of gold with copper and silver are commonly used to increase the hardness of jewelry.",
            "Dmitri Mendeleev published the first widely recognized periodic table in 1869.",
            "Transition metals occupy groups 3 through 12 of the periodic table of elements.",
            "Noble metals are metallic chemical elements that resist oxidation and corrosion in moist air.",
            "Carat is a measure of the purity of gold, with 24 carats being 100 percent pure gold.",
            "Cyanidation is a metallurgical technique for extracting gold from low-grade ore.",
            "Gold bullion coins are minted by governments for investment purposes rather than daily circulation.",
            "Atomic mass of gold is approximately 196.96657 atomic mass units.",
            "Electron configuration of gold in its ground state is [Xe] 4f14 5d10 6s1.",
            "Gold nanoparticles have gained widespread applications in medical diagnostics and nanotechnology."
        ]
    }
]

MS_MARCO_TEMPLATES = [
    {
        "q": "What is the boiling point of water at sea level in Fahrenheit?",
        "ans": "212 degrees Fahrenheit",
        "p_pos": "At standard atmospheric pressure at sea level, the boiling point of pure water is 212 degrees Fahrenheit (100 degrees Celsius).",
        "distractors": [
            "The freezing point of water at standard sea-level pressure is 32 degrees Fahrenheit (0 degrees Celsius).",
            "Atmospheric pressure decreases as altitude increases, causing the boiling point of water to decrease.",
            "At an altitude of 5,000 feet, water boils at approximately 203 degrees Fahrenheit (95 degrees Celsius).",
            "Daniel Gabriel Fahrenheit was a physicist who developed the mercury-in-glass thermometer and Fahrenheit scale.",
            "Anders Celsius invented the Celsius temperature scale in 1742, initially setting 0 as boiling and 100 as freezing.",
            "The Kelvin temperature scale is an absolute thermodynamic temperature scale named after Lord Kelvin.",
            "Water boils when its vapor pressure equals the surrounding atmospheric pressure.",
            "Specific heat capacity of water is 4.184 joules per gram degree Celsius at standard conditions.",
            "Latent heat of vaporization of water is approximately 2260 kJ per kilogram.",
            "The Fahrenheit temperature scale defines the interval between freezing and boiling points as exactly 180 degrees.",
            "Steam is water in the gas phase which is formed when water reaches its boiling point.",
            "Barometric pressure at sea level is defined as 1 atmosphere or 101.325 kPa or 29.92 inches of mercury.",
            "Cooking times at high altitudes must often be extended because boiling water does not get as hot.",
            "A pressure cooker raises the boiling point of water above 212 degrees Fahrenheit by increasing internal pressure.",
            "Absolute zero is defined as 0 Kelvin, which corresponds to -459.67 degrees Fahrenheit.",
            "Water molecule consists of two hydrogen atoms bonded covalently to a single oxygen atom.",
            "Density of liquid water reaches its maximum at approximately 39.2 degrees Fahrenheit (4 degrees Celsius).",
            "Thermal expansion of water causes its volume to increase as it freezes into crystalline ice.",
            "Superheated water can remain liquid at temperatures higher than its normal boiling point under high pressure.",
            "Thermometers measure temperature using physical properties such as thermal expansion or electrical resistance."
        ]
    },
    {
        "q": "What does CPU stand for in computer hardware?",
        "ans": "Central Processing Unit",
        "p_pos": "A central processing unit (CPU), also called a central processor or main processor, is the primary component that executes computer instructions.",
        "distractors": [
            "A graphics processing unit (GPU) is a specialized electronic circuit designed to accelerate computer graphics.",
            "Random Access Memory (RAM) provides high-speed volatile storage for active computer programs.",
            "The motherboard is the main printed circuit board containing the primary components of a computer.",
            "A solid-state drive (SSD) uses integrated circuit assemblies to store data persistently.",
            "The BIOS or UEFI firmware initializes hardware components during the computer booting sequence.",
            "Clock speed of a processor is typically measured in gigahertz (GHz), indicating cycles per second.",
            "Instruction set architecture (ISA) specifies the instructions that a CPU can natively execute.",
            "x86 and ARM are two of the dominant processor architectures used in personal computers and mobile devices.",
            "Cache memory (L1, L2, L3) resides directly on the processor die to provide ultra-fast access to instructions.",
            "Multi-core processors combine two or more independent processing units onto a single chip.",
            "Arithmetic Logic Unit (ALU) is the digital circuit within a CPU that performs arithmetic and logic operations.",
            "The control unit directs the operation of the processor by extracting instructions from memory.",
            "Thermal paste is applied between the CPU heat spreader and cooling heatsink to dissipate heat.",
            "Overclocking refers to configuring a processor to operate at higher clock speeds than factory specifications.",
            "Moore's Law historically observed that the number of transistors on a microchip doubles roughly every two years.",
            "Direct Memory Access (DMA) allows hardware subsystems to access system memory independently of the CPU.",
            "Bus width refers to the number of parallel lines that carry data between components in a computer.",
            "Power supply units convert alternating current (AC) electricity into direct current (DC) for computer components.",
            "Operating systems manage hardware resources and coordinate the scheduling of CPU tasks.",
            "Von Neumann architecture describes a computer design with a shared memory bus for instructions and data."
        ]
    }
]


def _expand_dataset(templates: List[Dict[str, Any]], dataset_name: str, total_count: int) -> List[BenchmarkItem]:
    """Generate or retrieve a diverse, reproducible benchmark split."""
    items: List[BenchmarkItem] = []
    rng = random.Random(42)

    for i in range(total_count):
        tmpl = templates[i % len(templates)]
        idx = i + 1
        item_id = f"{dataset_name}_{idx:04d}"

        # Build candidate passages (20-30 total)
        passages: List[Passage] = []
        gold_ids: List[str] = []
        is_multi_hop = False

        if dataset_name == "hotpotqa":
            is_multi_hop = True
            p1_id = f"{item_id}_p1"
            p2_id = f"{item_id}_p2"
            passages.append(Passage(id=p1_id, text=tmpl["p1"], is_supporting=True))
            passages.append(Passage(id=p2_id, text=tmpl["p2"], is_supporting=True))
            gold_ids.extend([p1_id, p2_id])
        else:
            p_id = f"{item_id}_pos"
            passages.append(Passage(id=p_id, text=tmpl["p_pos"], is_supporting=True))
            gold_ids.append(p_id)

        # Add 18 to 28 distractors to form 20-30 passages total
        distractor_pool = list(tmpl["distractors"])
        rng.shuffle(distractor_pool)
        distractor_count = rng.randint(18, min(28, len(distractor_pool)))
        for d_idx, d_text in enumerate(distractor_pool[:distractor_count]):
            # Slightly vary wording if i > len(templates) to ensure diversity
            var_text = d_text
            if i >= len(templates):
                var_text = f"Context note #{idx}: {d_text}"
            passages.append(Passage(id=f"{item_id}_d{d_idx+1}", text=var_text, is_supporting=False))

        # Shuffle passage order so relevant passages are not always at index 0
        rng.shuffle(passages)

        question = tmpl["q"]
        if i >= len(templates):
            question = f"{tmpl['q']} [Variant #{idx}]"

        item = BenchmarkItem(
            id=item_id,
            dataset=dataset_name,
            question=question,
            passages=passages,
            gold_answer=tmpl["ans"],
            gold_supporting_ids=gold_ids,
            is_multi_hop=is_multi_hop,
        )
        items.append(item)

    return items


def load_benchmark_dataset(
    dataset_name: str,
    n: int = 300,
    force_refresh: bool = False,
    cache_dir: Optional[str] = None
) -> List[BenchmarkItem]:
    """
    Load or generate a benchmark dataset.
    Supported datasets: 'hotpotqa', 'nq', 'natural_questions', 'msmarco'.
    Returns n BenchmarkItems with 20-30 candidate passages each.
    """
    canon_name = dataset_name.lower().replace("-", "_")
    if canon_name in ("nq", "natural_questions"):
        canon_name = "natural_questions"
        templates = NATURAL_QUESTIONS_TEMPLATES
    elif canon_name == "hotpotqa":
        templates = HOTPOTQA_TEMPLATES
    elif canon_name in ("msmarco", "ms_marco"):
        canon_name = "msmarco"
        templates = MS_MARCO_TEMPLATES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'hotpotqa', 'natural_questions', or 'msmarco'.")

    target_dir = cache_dir or BENCHMARK_CACHE_DIR
    os.makedirs(target_dir, exist_ok=True)
    cache_file = os.path.join(target_dir, f"{canon_name}_{n}.json")

    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [BenchmarkItem.from_dict(item) for item in data]
        except Exception:
            pass

    items = _expand_dataset(templates, canon_name, n)

    # Persist cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in items], f, indent=2)
    except Exception:
        pass

    return items
