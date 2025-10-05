"""
Improved LLM Extractor with chunking and retry logic
"""
import json
import re
import time
from typing import Dict, List, Optional, Tuple
import openai
from openai import OpenAI
from flask import current_app

class ImprovedLLMExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=current_app.config.get('OPENAI_API_KEY'))
        self.model = current_app.config.get('LLM_MODEL', 'gpt-4-turbo-preview')
        self.max_tokens = current_app.config.get('LLM_MAX_TOKENS', 4000)
        self.temperature = current_app.config.get('LLM_TEMPERATURE', 0.1)
        self.max_chunk_size = current_app.config.get('MAX_CHUNK_SIZE', 15000)
        self.tokens_used = 0
        self.finish_reason = None
    
    def extract_trial_data(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract trial data with intelligent chunking
        """
        # Check if text needs chunking
        if len(text) > self.max_chunk_size * 2:
            return self._extract_chunked(text, tables)
        else:
            return self._extract_single(text, tables)
    
    def _extract_chunked(self, text: str, tables: List[Dict]) -> Dict:
        """
        Extract data in chunks for long documents
        """
        print(f"Document too long ({len(text)} chars), using chunked extraction...")
        
        # Split document into sections
        sections = self._split_document_intelligently(text)
        
        # Extract from each section
        extracted_sections = {}
        for section_name, section_text in sections.items():
            if section_text:
                print(f"Extracting from section: {section_name} ({len(section_text)} chars)")
                section_data = self._extract_section(
                    section_name, 
                    section_text, 
                    tables if section_name == 'results' else None
                )
                extracted_sections[section_name] = section_data
        
        # Merge results
        merged_data = self._merge_section_extractions(extracted_sections)
        return merged_data
    
    def _split_document_intelligently(self, text: str) -> Dict[str, str]:
        """
        Split document into semantic sections
        """
        sections = {
            'header': '',
            'methods': '',
            'results': '',
            'interventions': '',
            'population': '',
            'outcomes': '',
            'statistics': '',
            'discussion': ''
        }
        
        # Define section patterns with multiple possible headers
        section_patterns = {
            'methods': [
                r'(?i)\n\s*(methods|methodology|study design|materials and methods)\s*\n',
                r'(?i)\n\s*(experimental (design|procedures?))\s*\n'
            ],
            'results': [
                r'(?i)\n\s*(results|findings|outcomes)\s*\n',
                r'(?i)\n\s*(primary (outcome|endpoint)s?)\s*\n'
            ],
            'interventions': [
                r'(?i)\n\s*(intervention|treatment|randomization)\s*\n',
                r'(?i)\n\s*(study (drug|medication|treatment))\s*\n'
            ],
            'population': [
                r'(?i)\n\s*(participants?|patients?|subjects?|population)\s*\n',
                r'(?i)\n\s*(eligibility|inclusion|exclusion)\s*\n'
            ],
            'statistics': [
                r'(?i)\n\s*(statistical (analysis|methods?))\s*\n',
                r'(?i)\n\s*(sample size|power calculation)\s*\n'
            ],
            'discussion': [
                r'(?i)\n\s*(discussion|conclusions?)\s*\n'
            ]
        }
        
        # Extract header (first 3000 chars usually contains title, authors, abstract)
        sections['header'] = text[:3000]
        
        # Find and extract each section
        used_positions = set()
        for section_name, patterns in section_patterns.items():
            best_match = None
            best_position = len(text)
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match and match.start() not in used_positions:
                    if match.start() < best_position:
                        best_match = match
                        best_position = match.start()
            
            if best_match:
                used_positions.add(best_position)
                start = best_position
                
                # Find the end of this section (start of next section or end of document)
                end = len(text)
                for other_section, other_patterns in section_patterns.items():
                    if other_section != section_name:
                        for pattern in other_patterns:
                            next_match = re.search(pattern, text[start + 100:])
                            if next_match:
                                potential_end = start + 100 + next_match.start()
                                if potential_end < end:
                                    end = potential_end
                
                # Limit section size
                section_text = text[start:min(end, start + self.max_chunk_size)]
                sections[section_name] = section_text
        
        # Remove empty sections
        sections = {k: v for k, v in sections.items() if v.strip()}
        
        return sections
    
    def _extract_section(self, section_name: str, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract data from a specific section with focused prompt
        """
        # Create section-specific prompts
        section_prompts = {
            'header': self._create_header_prompt,
            'methods': self._create_methods_prompt,
            'results': self._create_results_prompt,
            'interventions': self._create_interventions_prompt,
            'population': self._create_population_prompt,
            'outcomes': self._create_outcomes_prompt,
            'statistics': self._create_statistics_prompt
        }
        
        prompt_creator = section_prompts.get(section_name, self._create_generic_prompt)
        prompt = prompt_creator(text, tables)
        
        return self._call_llm_with_retry(prompt)
    
    def _extract_single(self, text: str, tables: List[Dict]) -> Dict:
        """
        Extract from entire document at once
        """
        prompt = self._create_comprehensive_prompt(text[:self.max_chunk_size * 2], tables)
        return self._call_llm_with_retry(prompt)
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Dict:
        """
        Call LLM with exponential backoff retry
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert clinical trial data extractor with 20+ years of experience. Extract data precisely, never hallucinate, and mark unclear data as 'NOT_REPORTED'."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                self.tokens_used += response.usage.total_tokens
                self.finish_reason = response.choices[0].finish_reason
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            except openai.RateLimitError:
                wait_time = 2 ** attempt
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {}
            except Exception as e:
                print(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {}
        
        return {}
    
    def _merge_section_extractions(self, sections: Dict[str, Dict]) -> Dict:
        """
        Intelligently merge extracted sections
        """
        merged = {
            "study_identification": {},
            "study_design": {},
            "population": {},
            "interventions": [],
            "outcomes": {"primary": [], "secondary": []},
            "subgroup_analyses": [],
            "adverse_events": [],
            "statistical_methods": {},
            "risk_of_bias": {},
            "funding_conflicts": {},
            "data_extraction_notes": {}
        }
        
        # Merge each section's data
        for section_name, section_data in sections.items():
            if not section_data:
                continue
            
            # Merge study identification (prefer from header)
            if section_name == 'header' and 'study_identification' in section_data:
                merged['study_identification'].update(section_data['study_identification'])
            elif 'study_identification' in section_data:
                for key, value in section_data['study_identification'].items():
                    if key not in merged['study_identification'] or not merged['study_identification'][key]:
                        merged['study_identification'][key] = value
            
            # Merge study design (prefer from methods)
            if section_name == 'methods' and 'study_design' in section_data:
                merged['study_design'].update(section_data['study_design'])
            elif 'study_design' in section_data:
                for key, value in section_data['study_design'].items():
                    if key not in merged['study_design'] or not merged['study_design'][key]:
                        merged['study_design'][key] = value
            
            # Merge population
            if 'population' in section_data:
                merged['population'].update(section_data['population'])
            
            # Merge interventions (avoid duplicates)
            if 'interventions' in section_data:
                for intervention in section_data['interventions']:
                    if intervention and not any(
                        i.get('arm_name') == intervention.get('arm_name') 
                        for i in merged['interventions']
                    ):
                        merged['interventions'].append(intervention)
            
            # Merge outcomes
            if 'outcomes' in section_data:
                if 'primary' in section_data['outcomes']:
                    merged['outcomes']['primary'].extend(section_data['outcomes']['primary'])
                if 'secondary' in section_data['outcomes']:
                    merged['outcomes']['secondary'].extend(section_data['outcomes']['secondary'])
            
            # Merge other fields
            for field in ['subgroup_analyses', 'adverse_events']:
                if field in section_data:
                    merged[field].extend(section_data[field])
            
            for field in ['statistical_methods', 'risk_of_bias', 'funding_conflicts']:
                if field in section_data:
                    merged[field].update(section_data[field])
        
        return merged
    
    # Prompt creation methods
    def _create_header_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return f"""Extract study identification from this clinical trial header section.

Return JSON with:
{{
  "study_identification": {{
    "title": "complete title",
    "authors": ["all authors"],
    "journal": "journal name",
    "year": "publication year",
    "doi": "DOI if present",
    "trial_registration": "NCT or other registration number"
  }}
}}

Text:
{text}

Extract the information. Use "NOT_REPORTED" for missing data."""

    def _create_methods_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return f"""Extract study design and methodology from this clinical trial methods section.

Return JSON with:
{{
  "study_design": {{
    "type": "study design type (e.g., parallel-group RCT)",
    "blinding": "blinding method",
    "randomization_method": "randomization description",
    "allocation_ratio": "e.g., 1:1",
    "duration_treatment": "treatment duration",
    "number_of_sites": "number of sites",
    "country": "countries involved"
  }},
  "population": {{
    "inclusion_criteria": ["list criteria"],
    "exclusion_criteria": ["list criteria"]
  }}
}}

Methods text:
{text}

Extract all available information. Use "NOT_REPORTED" for missing data."""

    def _create_results_prompt(self, text: str, tables: List[Dict] = None) -> str:
        table_context = ""
        if tables:
            table_context = "\n\nExtracted tables:\n" + json.dumps(tables[:5], indent=2)[:3000]
        
        return f"""Extract results and outcomes from this clinical trial results section.

Return JSON with:
{{
  "outcomes": {{
    "primary": [
      {{
        "outcome_name": "name",
        "timepoint": "when measured",
        "results_by_arm": [
          {{
            "arm": "arm name",
            "n": "number analyzed",
            "mean": "mean value",
            "sd": "standard deviation",
            "events": "for binary outcomes",
            "total": "denominator"
          }}
        ],
        "between_group_comparison": {{
          "effect_measure": "OR/RR/MD/etc",
          "effect_estimate": "value",
          "ci_95_lower": "lower CI",
          "ci_95_upper": "upper CI",
          "p_value": "p-value"
        }}
      }}
    ],
    "secondary": []
  }},
  "adverse_events": [
    {{
      "event_name": "event",
      "severity": "severity",
      "results_by_arm": []
    }}
  ]
}}

Results text:
{text}
{table_context}

Extract all numerical results. Use exact values from tables when available."""

    def _create_comprehensive_prompt(self, text: str, tables: List[Dict] = None) -> str:
        """Create comprehensive prompt for full document extraction"""
        table_context = ""
        if tables:
            table_context = f"\n\nExtracted tables ({len(tables)} total):\n"
            table_context += json.dumps(tables[:5], indent=2)[:5000]
        
        return f"""Extract ALL data from this clinical trial for meta-analysis.

Return complete JSON with ALL sections:
- study_identification (title, authors, registration)
- study_design (type, blinding, randomization)
- population (criteria, demographics)
- interventions (all arms with doses)
- outcomes (primary and secondary with statistics)
- subgroup_analyses
- adverse_events
- statistical_methods
- risk_of_bias

Clinical trial text:
{text}
{table_context}

Extract comprehensively. Mark missing data as "NOT_REPORTED"."""

    def _create_generic_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return self._create_comprehensive_prompt(text, tables)

    def _create_interventions_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return f"""Extract intervention details from this clinical trial section.

Return JSON with:
{{
  "interventions": [
    {{
      "arm_name": "descriptive name",
      "n_randomized": "number randomized",
      "drug_name": "drug name",
      "dose": "dosage",
      "frequency": "dosing schedule",
      "duration": "treatment duration"
    }}
  ]
}}

Text:
{text}

Extract all intervention arms with complete details."""

    def _create_population_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return f"""Extract population and baseline characteristics from this clinical trial section.

Return JSON with:
{{
  "population": {{
    "total_screened": "number",
    "total_randomized": "number",
    "age_mean": "mean age",
    "age_sd": "SD",
    "sex_male_percent": "percent male",
    "inclusion_criteria": ["criteria"],
    "exclusion_criteria": ["criteria"],
    "baseline_characteristics": {{}}
  }}
}}

Text:
{text}

Extract all demographic and baseline data."""

    def _create_outcomes_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return self._create_results_prompt(text, tables)

    def _create_statistics_prompt(self, text: str, tables: List[Dict] = None) -> str:
        return f"""Extract statistical methods from this clinical trial section.

Return JSON with:
{{
  "statistical_methods": {{
    "primary_analysis_method": "method",
    "missing_data_method": "method",
    "adjustment_variables": ["variables"],
    "sample_size_calculation": "calculation details"
  }}
}}

Text:
{text}

Extract all statistical methodology details."""