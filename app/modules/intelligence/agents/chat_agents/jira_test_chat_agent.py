import json
import os
from typing import Dict, List, AsyncGenerator, Any

from crewai import Agent, Crew, Process, Task
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.provider.provider_service import AgentType
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import get_code_from_node_id_tool
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import get_code_from_probable_node_name_tool
from app.modules.intelligence.tools.web_tools.github_tool import github_tool


class JiraTestChatAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = llm  # This will be CrewAI LLM
        self.db = db
        self.max_iterations = os.getenv("MAX_ITER", 15)
        
        # Initialize tools
        self.get_code_from_node_id = get_code_from_node_id_tool(db, None)  # user_id not needed for tools
        self.get_code_from_probable_node_name = get_code_from_probable_node_name_tool(db, None)
        if os.getenv("GITHUB_APP_ID"):
            self.github_tool = github_tool(db, None)

    async def create_agents(self):
        # Define common agent parameters with explicit provider configuration
        common_agent_params = {
            'allow_delegation': True,
            'llm': "ollama_chat/deepseek-r1:7b",
            # {
            #     'model': 'deepseek-r1:7b',  # Use the model name directly
            #     'provider': 'ollama',  # Explicitly set provider
            #     'config': {
            #         'temperature': 0.3,
            #         'max_tokens': 4096,
            #         'top_k': 50,
            #         'top_p': 0.9,
            #         'repeat_penalty': 1.1,
            #     }
            # },
            'max_iterations': self.max_iterations,
            'verbose': True
        }
        
        # Create the test plan creator agent
        test_planner = Agent(
            role="Test Plan Creator",
            goal="Analyze code changes and create comprehensive test plans in Jira format",
            backstory="""You are an expert test planner who specializes in analyzing code changes 
            and creating detailed test plans. You focus on identifying critical test scenarios 
            based on recent code modifications.
            
            IMPORTANT: You must ALWAYS format your responses as valid JSON with this structure:
            {
                \"message\": \"Your actual response content here\",
                \"citations\": []  // Always an empty array
            }
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_probable_node_name,
            ] + ([self.github_tool] if hasattr(self, 'github_tool') else []),
            **common_agent_params
        )

        # Create the Jira test case writer agent
        test_writer = Agent(
            role="Test Case Writer",
            goal="Write detailed test cases following Jira's structured format",
            backstory="""You are a skilled test case writer who excels at creating clear, 
            actionable test cases in Jira format. You ensure test cases are detailed, 
            reproducible, and cover all edge cases.
            
            IMPORTANT: You must ALWAYS format your responses as valid JSON with this structure:
            {
                \"message\": \"Your actual response content here\",
                \"citations\": []  // Always an empty array
            }
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_probable_node_name,
            ],
            **common_agent_params
        )

        return [test_planner, test_writer]

    async def create_tasks(self, agents, query: str, project_id: str, node_ids: List[NodeContext]):
        [test_planner, test_writer] = agents

        # Task 1: Analyze code changes and create test plan
        analyze_changes = Task(
            description=f"""
            1. Use the github_tool to fetch recent commits and code changes
            2. Analyze the changes to identify areas requiring testing
            3. Create a test plan that covers:
               - Unit tests for modified functions
               - Integration tests for affected components
               - Edge cases and potential failure scenarios
            4. Organize test scenarios by priority
            
            Format your response as valid JSON with this schema:
            {{
                "message": "Your detailed test plan here",
                "citations": []  // Leave this empty array
            }}
            
            Context:
            - Project ID: {project_id}
            - Query: {query}
            """,
            agent=test_planner,
            expected_output="A comprehensive test plan in JSON format analyzing code changes and identifying test scenarios"
        )

        # Task 2: Write Jira test cases
        write_tests = Task(
            description=f"""
            Format your response as valid JSON with this schema:
            {{
                "message": "Your test cases in Jira format as shown below",
                "citations": []  // Leave this empty array
            }}
            
            For the message content, use this Jira formatting:
            
            h2. Test Case: [Test case title]
            *Description:* [Test objective]
            
            h3. Preconditions
            * [List preconditions]
            
            h3. Test Steps
            # [Step 1 description]
            ** Expected: [Expected result]
            # [Step 2 description]
            ** Expected: [Expected result]
            
            h3. Test Data
            * [Required test data]
            
            h3. Priority
            {{color:red}}*High*{{color}} or {{color:orange}}*Medium*{{color}} or {{color:green}}*Low*{{color}}
            
            h3. Components
            * [Component 1]
            * [Component 2]
            
            Context:
            - Project ID: {project_id}
            - Query: {query}
            """,
            agent=test_writer,
            expected_output="Detailed test cases in Jira format"
        )

        return [analyze_changes, write_tests]

    async def run(self, query: str, project_id: str, node_ids: List[NodeContext]) -> AsyncGenerator[str, None]:
        try:
            # Create agents
            agents = await self.create_agents()
            
            # Create tasks
            tasks = await self.create_tasks(agents, query, project_id, node_ids)
            
            # Create crew with explicit provider configuration
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential,
                llm=self.llm  # Pass the configured LLM instance
            )

            # Initialize response buffer
            response_buffer = ""
            
            # Execute tasks and get result
            result = crew.kickoff()
            
            # Get the final output from the crew
            final_output = str(result)
            
            # Try to parse the output as JSON
            try:
                parsed_json = json.loads(final_output)
                if isinstance(parsed_json, dict):
                    content = parsed_json.get("message", final_output)
                else:
                    content = final_output
            except json.JSONDecodeError:
                content = final_output
            
            # Yield the final response
            yield json.dumps({
                "message": content,
                "citations": []
            })

        except Exception as e:
            yield json.dumps({
                "message": f"Error running Jira test agent: {str(e)}",
                "citations": []
            })
