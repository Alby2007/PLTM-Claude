"""
Multi-Agent Collaboration with Shared Memory

Enables multiple agents to share a memory system for collective intelligence.
This is OPTIONAL and does not modify the core memory system.

Key features:
- Multiple agents share workspace memory
- Agents build on each other's work
- Collective knowledge accumulation
- Agent coordination without complex protocols

Research potential: "Emergent Collaboration in Multi-Agent Systems via Shared Memory"
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from loguru import logger

from src.pipeline.memory_pipeline import MemoryPipeline


@dataclass
class AgentProfile:
    """Profile for an agent in the workspace"""
    agent_id: str
    role: str
    capabilities: List[str]
    contributions: List[str]
    joined_at: datetime


class SharedMemoryWorkspace:
    """
    Workspace where multiple agents share memory for collaboration.
    
    Example use case:
        - Agent A (Research): Learns about topic X
        - Agent B (Writing): Can access what A learned
        - Agent C (Coding): References both A and B's knowledge
        
    Result: Agents build on each other's work without explicit communication.
    """
    
    def __init__(
        self,
        memory_pipeline: MemoryPipeline,
        workspace_id: str,
        workspace_name: str = "Untitled Workspace"
    ):
        """
        Initialize shared workspace.
        
        Args:
            memory_pipeline: Existing memory pipeline (no modifications)
            workspace_id: Unique workspace identifier
            workspace_name: Human-readable workspace name
        """
        self.pipeline = memory_pipeline
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name
        self.agents: Dict[str, AgentProfile] = {}
        
        # Stats
        self.total_contributions = 0
        self.total_retrievals = 0
        
        logger.info(
            f"SharedMemoryWorkspace '{workspace_name}' initialized "
            f"(workspace_id={workspace_id})"
        )
    
    async def add_agent(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str]
    ) -> AgentProfile:
        """
        Register agent in workspace.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent's role (e.g., "Researcher", "Writer", "Coder")
            capabilities: List of agent capabilities
            
        Returns:
            Agent profile
        """
        profile = AgentProfile(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            contributions=[],
            joined_at=datetime.now()
        )
        
        self.agents[agent_id] = profile
        
        # Store agent metadata in shared memory
        await self.pipeline.process(
            user_id=f"workspace_{self.workspace_id}",
            message=f"Agent {agent_id} joined with role: {role}. Capabilities: {', '.join(capabilities)}"
        )
        
        logger.info(f"Agent {agent_id} ({role}) joined workspace {self.workspace_id}")
        
        return profile
    
    async def agent_contribute(
        self,
        agent_id: str,
        contribution: str,
        contribution_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Agent makes a contribution to shared knowledge.
        
        Args:
            agent_id: Agent making the contribution
            contribution: The knowledge/work being contributed
            contribution_type: Type of contribution (research, code, analysis, etc.)
            
        Returns:
            Contribution metadata
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered in workspace")
        
        # Store contribution in shared memory
        await self.pipeline.process(
            user_id=f"workspace_{self.workspace_id}",
            message=f"[{self.agents[agent_id].role}] {contribution}"
        )
        
        # Track contribution
        self.agents[agent_id].contributions.append(contribution)
        self.total_contributions += 1
        
        metadata = {
            "agent_id": agent_id,
            "role": self.agents[agent_id].role,
            "contribution_type": contribution_type,
            "timestamp": datetime.now().isoformat(),
            "contribution_number": len(self.agents[agent_id].contributions)
        }
        
        logger.info(
            f"Agent {agent_id} contributed ({contribution_type}): "
            f"{contribution[:100]}..."
        )
        
        return metadata
    
    async def get_shared_context(
        self,
        query: str,
        limit: int = 50,
        exclude_agent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve shared knowledge relevant to query.
        
        Args:
            query: What to search for
            limit: Maximum memories to retrieve
            exclude_agent: Optionally exclude specific agent's contributions
            
        Returns:
            List of relevant memories from all agents
        """
        self.total_retrievals += 1
        
        # Get all workspace memories
        atoms = await self.pipeline.store.find_by_triple(
            subject=f"workspace_{self.workspace_id}",
            exclude_historical=True
        )
        
        # Format for consumption
        context = []
        for atom in atoms[:limit]:
            # Parse agent from object if possible
            agent_info = "Unknown"
            for agent_id, profile in self.agents.items():
                if f"[{profile.role}]" in atom.object:
                    agent_info = f"{agent_id} ({profile.role})"
                    break
            
            context.append({
                "content": atom.object,
                "agent": agent_info,
                "confidence": atom.confidence,
                "timestamp": atom.first_observed.isoformat()
            })
        
        logger.debug(
            f"Retrieved {len(context)} shared memories for query: {query[:50]}..."
        )
        
        return context
    
    async def agent_task(
        self,
        agent_id: str,
        task: str,
        use_shared_context: bool = True
    ) -> Dict[str, Any]:
        """
        Agent performs task with access to shared memory.
        
        Args:
            agent_id: Agent performing the task
            task: Task description
            use_shared_context: Whether to retrieve shared context
            
        Returns:
            Task result with metadata
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered in workspace")
        
        # Get shared context if enabled
        shared_context = []
        if use_shared_context:
            shared_context = await self.get_shared_context(
                query=task,
                limit=30,
                exclude_agent=agent_id
            )
        
        # In a real implementation, you'd call an LLM here with the context
        # For now, we return the structure for the experiment
        
        result = {
            "agent_id": agent_id,
            "role": self.agents[agent_id].role,
            "task": task,
            "shared_context_used": len(shared_context),
            "context": shared_context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Agent {agent_id} executing task with "
            f"{len(shared_context)} shared memories"
        )
        
        return result
    
    def get_workspace_stats(self) -> Dict[str, Any]:
        """Get workspace statistics"""
        return {
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "total_agents": len(self.agents),
            "total_contributions": self.total_contributions,
            "total_retrievals": self.total_retrievals,
            "agents": {
                agent_id: {
                    "role": profile.role,
                    "capabilities": profile.capabilities,
                    "contributions_count": len(profile.contributions),
                    "joined_at": profile.joined_at.isoformat()
                }
                for agent_id, profile in self.agents.items()
            }
        }


class MultiAgentExperiment:
    """
    Experiment framework for measuring multi-agent collaboration effectiveness.
    
    Hypothesis: Agents with shared memory collaborate more effectively than
    agents working independently.
    
    Metrics:
    - Time to task completion
    - Code quality / output quality
    - Need for human intervention
    - Knowledge reuse rate
    """
    
    def __init__(
        self,
        workspace: SharedMemoryWorkspace,
        experiment_name: str = "Multi-Agent Collaboration"
    ):
        """
        Initialize experiment.
        
        Args:
            workspace: Shared memory workspace
            experiment_name: Name for this experiment
        """
        self.workspace = workspace
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Metrics
        self.tasks_completed = 0
        self.knowledge_reuses = 0
        self.collaboration_events = 0
        
        logger.info(f"MultiAgentExperiment '{experiment_name}' initialized")
    
    async def run_collaborative_task(
        self,
        task_description: str,
        agent_sequence: List[str]
    ) -> Dict[str, Any]:
        """
        Run a task that requires multiple agents to collaborate.
        
        Args:
            task_description: Overall task description
            agent_sequence: Ordered list of agent IDs to work on task
            
        Returns:
            Task results with collaboration metrics
        """
        results = []
        
        for i, agent_id in enumerate(agent_sequence):
            # Each agent builds on previous agents' work
            subtask = f"{task_description} (step {i+1}/{len(agent_sequence)})"
            
            result = await self.workspace.agent_task(
                agent_id=agent_id,
                task=subtask,
                use_shared_context=True
            )
            
            results.append(result)
            
            # Track collaboration
            if result["shared_context_used"] > 0:
                self.collaboration_events += 1
                self.knowledge_reuses += result["shared_context_used"]
        
        self.tasks_completed += 1
        
        return {
            "task": task_description,
            "agents_involved": len(agent_sequence),
            "total_collaboration_events": self.collaboration_events,
            "knowledge_reuses": self.knowledge_reuses,
            "results": results
        }
    
    def get_experiment_results(self) -> Dict[str, Any]:
        """Get experiment results"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration,
            "tasks_completed": self.tasks_completed,
            "collaboration_events": self.collaboration_events,
            "knowledge_reuses": self.knowledge_reuses,
            "avg_reuses_per_task": (
                self.knowledge_reuses / self.tasks_completed
                if self.tasks_completed > 0 else 0
            ),
            "workspace_stats": self.workspace.get_workspace_stats()
        }
