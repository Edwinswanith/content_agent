from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from .tools.perplexitysearch import PerplexitySearchTool
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini-2024-07-18",              
    api_key="your_openai_api_key_here",
    temperature=0.7
)


@CrewBase
class ContentAgent():
	"""ContentAgent crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def research_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['research_agent'],
			tools=[SerperDevTool(), PerplexitySearchTool()],
			llm=llm,
			verbose=True
		)

	@agent
	def consolidation_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['consolidation_agent'],
			llm=llm,
			verbose=True
		)

	@agent
	def content_creation_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['content_creation_agent'],
			llm=llm,
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_gathering_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_gathering_task'],
			llm=llm,
		)

	@task
	def consolidation_task(self) -> Task:
		return Task(
			config=self.tasks_config['consolidation_task'],
			llm=llm,
		)

	@task
	def content_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['content_creation_task'],
			output_file='report.md',
			llm=llm,
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ContentAgent crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
