"""
MCP RAG integration with caching and error handling
"""
import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from cachetools import TTLCache
import aiohttp

from app.models.response_models import ErrorResponse, SuccessResponse


@dataclass
class RAGResponse:
    """RAG response model"""
    success: bool
    content: str
    sources: List[str]
    processing_time: float
    cached: bool
    language: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class RAGQuery:
    """RAG query model"""
    query: str
    language: str
    service: str  # tabib, f16, digimaman
    context: Optional[str] = None
    max_length: int = 1000


class QueryOptimizer:
    """Query enhancement for better RAG results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Persian query enhancement patterns
        self.persian_expansions = {
            'دیابت': 'دیابت قند خون',
            'فشار': 'فشار خون',
            'کودک': 'کودک کودکان',
            'مادر': 'مادر مادران والدین',
            'درمان': 'درمان معالجه',
            'علائم': 'علائم نشانه ها',
        }
        
        # Business query expansions
        self.business_expansions = {
            'استارتاپ': 'استارتاپ نوپا کارآفرینی',
            'سرمایه': 'سرمایه گذاری مالی',
            'بازار': 'بازار فروش',
            'مشتری': 'مشتری مخاطب',
            'محصول': 'محصول خدمات',
        }
    
    def enhance_query(self, query: str, service: str, language: str) -> str:
        """Enhance query for better RAG results"""
        try:
            enhanced_query = query.strip()
            
            # Service-specific optimizations
            if service == "tabib":
                enhanced_query = self._enhance_medical_query(enhanced_query, language)
            elif service == "f16":
                enhanced_query = self._enhance_business_query(enhanced_query, language)
            elif service == "digimaman":
                enhanced_query = self._enhance_parenting_query(enhanced_query, language)
            
            # General query improvements
            enhanced_query = self._add_context_keywords(enhanced_query, service)
            enhanced_query = self._optimize_for_persian(enhanced_query, language)
            
            self.logger.debug(f"📝 Query enhanced: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"❌ Query enhancement failed: {str(e)}")
            return query  # Return original on failure
    
    def _enhance_medical_query(self, query: str, language: str) -> str:
        """Enhance medical queries for Tabib service"""
        if language == "fa":
            # Add medical context
            for term, expansion in self.persian_expansions.items():
                if term in query:
                    query = query.replace(term, expansion)
            
            # Add common medical prefixes
            if not any(prefix in query for prefix in ['علائم', 'درمان', 'پیشگیری', 'علت']):
                query = f"علائم و درمان {query}"
        
        return query
    
    def _enhance_business_query(self, query: str, language: str) -> str:
        """Enhance business queries for F16 service"""
        if language == "fa":
            # Add business context
            for term, expansion in self.business_expansions.items():
                if term in query:
                    query = query.replace(term, expansion)
            
            # Add entrepreneurship context
            if not any(prefix in query for prefix in ['کارآفرینی', 'استارتاپ', 'کسب و کار']):
                query = f"کارآفرینی و {query}"
        
        return query
    
    def _enhance_parenting_query(self, query: str, language: str) -> str:
        """Enhance parenting queries for Digimaman service"""
        if language == "fa":
            # Add parenting context
            if not any(prefix in query for prefix in ['کودک', 'فرزند', 'والدین', 'تربیت']):
                query = f"تربیت کودک و {query}"
        
        return query
    
    def _add_context_keywords(self, query: str, service: str) -> str:
        """Add service-specific context keywords"""
        context_map = {
            "tabib": "سلامت پزشکی",
            "f16": "کسب و کار",
            "digimaman": "تربیت کودک"
        }
        
        context = context_map.get(service, "")
        if context and context not in query:
            query = f"{context} {query}"
        
        return query
    
    def _optimize_for_persian(self, query: str, language: str) -> str:
        """Optimize query for Persian language processing"""
        if language == "fa":
            # Normalize Persian characters
            query = query.replace('ي', 'ی').replace('ك', 'ک')
            
            # Add question words if missing
            if not any(word in query for word in ['چه', 'چگونه', 'چرا', 'کدام', 'کی']):
                if '؟' not in query:
                    query = f"چه {query}"
        
        return query


class MCPClient:
    """MCP service client for RAG operations"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Service endpoints
        self.endpoints = {
            "tabib": "tabib",
            "f16": "f16", 
            "digimaman": "digimaman",
            "time": "get_current_time"
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"}
        )
        self.logger.info("✅ MCP Client initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def query_service(self, service: str, query: str) -> Dict[str, Any]:
        """Query MCP service with error handling"""
        if not self.session:
            await self.initialize()
        
        try:
            endpoint = self.endpoints.get(service)
            if not endpoint:
                raise ValueError(f"Unknown service: {service}")
            
            # Prepare MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": endpoint,
                    "arguments": {"query": query} if service != "time" else {}
                },
                "id": str(int(time.time()))
            }
            
            self.logger.debug(f"🔗 Querying MCP service {service}: {query[:100]}...")
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=mcp_request
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"MCP request failed with status {response.status}")
                
                result = await response.json()
                
                if "error" in result:
                    raise Exception(f"MCP error: {result['error']}")
                
                return result.get("result", {})
                
        except Exception as e:
            self.logger.error(f"❌ MCP query failed for {service}: {str(e)}")
            raise


class CachedRAGService:
    """MCP RAG integration with caching and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mcp_client: Optional[MCPClient] = None
        self.response_cache = TTLCache(maxsize=200, ttl=1800)  # 30 minutes
        self.query_optimizer = QueryOptimizer()
        
        # Fallback responses for common queries
        self.fallback_responses = {
            "health_general": "برای مشاوره پزشکی دقیق، لطفاً با پزشک مراجعه کنید.",
            "business_general": "برای راهنمایی کسب و کار، مطالعه منابع معتبر توصیه می‌شود.",
            "parenting_general": "در تربیت کودک، صبر و محبت اساسی است."
        }
        
        # Performance metrics
        self.total_queries = 0
        self.cache_hits = 0
        self.errors = 0
        self.total_response_time = 0.0
        
        self.logger.info("🧠 Cached RAG Service created")
    
    async def initialize(self, mcp_base_url: str):
        """Initialize MCP client"""
        try:
            self.mcp_client = MCPClient(mcp_base_url)
            await self.mcp_client.initialize()
            
            # Test connection
            await self._test_mcp_connection()
            
            self.logger.info("✅ RAG Service initialized with MCP")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize RAG service: {str(e)}")
            raise
    
    async def query_tabib_with_cache(self, query: str, language: str = "fa") -> RAGResponse:
        """Retrieve medical knowledge with intelligent caching"""
        return await self._query_service_with_cache("tabib", query, language)
    
    async def query_f16_with_cache(self, query: str, language: str = "fa") -> RAGResponse:
        """Retrieve business knowledge with intelligent caching"""
        return await self._query_service_with_cache("f16", query, language)
    
    async def query_digimaman_with_cache(self, query: str, language: str = "fa") -> RAGResponse:
        """Retrieve parenting knowledge with intelligent caching"""
        return await self._query_service_with_cache("digimaman", query, language)
    
    async def _query_service_with_cache(self, service: str, query: str, language: str) -> RAGResponse:
        """Generic service query with caching"""
        start_time = time.time()
        
        try:
            self.total_queries += 1
            
            # Check cache for similar queries
            cache_key = self._generate_cache_key(service, query, language)
            
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                self.cache_hits += 1
                processing_time = time.time() - start_time
                
                self.logger.debug(f"📦 Cache hit for {service} query")
                
                return RAGResponse(
                    success=True,
                    content=cached_response["content"],
                    sources=cached_response.get("sources", []),
                    processing_time=processing_time,
                    cached=True,
                    language=language,
                    confidence=cached_response.get("confidence", 0.9),
                    metadata=cached_response.get("metadata", {})
                )
            
            # Optimize query for better RAG results
            optimized_query = self.query_optimizer.enhance_query(query, service, language)
            
            # Query MCP service with error handling
            try:
                mcp_result = await self.mcp_client.query_service(service, optimized_query)
                response_content = self._parse_mcp_response(mcp_result, service)
                
                # Cache successful response
                cache_data = {
                    "content": response_content["content"],
                    "sources": response_content.get("sources", []),
                    "confidence": response_content.get("confidence", 0.8),
                    "metadata": response_content.get("metadata", {})
                }
                self.response_cache[cache_key] = cache_data
                
                processing_time = time.time() - start_time
                self.total_response_time += processing_time
                
                return RAGResponse(
                    success=True,
                    content=response_content["content"],
                    sources=response_content.get("sources", []),
                    processing_time=processing_time,
                    cached=False,
                    language=language,
                    confidence=response_content.get("confidence", 0.8),
                    metadata=response_content.get("metadata", {})
                )
                
            except Exception as mcp_error:
                self.logger.warning(f"⚠️ MCP query failed, using fallback: {str(mcp_error)}")
                fallback_response = await self._handle_rag_service_failure(service, query, mcp_error)
                
                processing_time = time.time() - start_time
                return fallback_response
                
        except Exception as e:
            self.errors += 1
            processing_time = time.time() - start_time
            self.logger.error(f"❌ RAG query failed for {service}: {str(e)}")
            
            return RAGResponse(
                success=False,
                content=f"خطا در دریافت اطلاعات: {str(e)}",
                sources=[],
                processing_time=processing_time,
                cached=False,
                language=language,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def _handle_rag_service_failure(self, service: str, query: str, error: Exception) -> RAGResponse:
        """Graceful degradation when RAG service fails"""
        try:
            self.logger.info(f"🔄 Handling {service} service failure gracefully")
            
            # Check for cached responses to similar queries
            similar_response = self._find_similar_cached_response(service, query)
            if similar_response:
                self.logger.info("📦 Using similar cached response as fallback")
                return RAGResponse(
                    success=True,
                    content=similar_response["content"],
                    sources=similar_response.get("sources", []),
                    processing_time=0.1,
                    cached=True,
                    language="fa",
                    confidence=0.6,  # Lower confidence for fallback
                    metadata={"fallback": True, "original_error": str(error)}
                )
            
            # Generate appropriate fallback response
            fallback_content = self._generate_fallback_response(service, query)
            
            return RAGResponse(
                success=True,
                content=fallback_content,
                sources=["fallback"],
                processing_time=0.05,
                cached=False,
                language="fa",
                confidence=0.3,  # Low confidence for generic fallback
                metadata={"fallback": True, "error": str(error)}
            )
            
        except Exception as fallback_error:
            self.logger.error(f"❌ Fallback handling failed: {str(fallback_error)}")
            return RAGResponse(
                success=False,
                content="متأسفانه در حال حاضر امکان پاسخگویی وجود ندارد.",
                sources=[],
                processing_time=0.01,
                cached=False,
                language="fa",
                confidence=0.0,
                metadata={"error": str(fallback_error)}
            )
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get RAG service performance metrics"""
        avg_response_time = 0.0
        if self.total_queries > 0:
            avg_response_time = self.total_response_time / self.total_queries
        
        cache_hit_rate = 0.0
        if self.total_queries > 0:
            cache_hit_rate = self.cache_hits / self.total_queries
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.total_queries),
            "average_response_time": avg_response_time,
            "cache_size": len(self.response_cache),
            "mcp_status": "connected" if self.mcp_client else "disconnected"
        }
    
    async def _test_mcp_connection(self):
        """Test MCP connection"""
        try:
            result = await self.mcp_client.query_service("time", "")
            self.logger.info("✅ MCP connection test successful")
            
        except Exception as e:
            self.logger.error(f"❌ MCP connection test failed: {str(e)}")
            raise
    
    def _parse_mcp_response(self, mcp_result: Dict[str, Any], service: str) -> Dict[str, Any]:
        """Parse MCP response based on service type"""
        try:
            content = mcp_result.get("content", [])
            
            if isinstance(content, list) and content:
                # Extract text content
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                
                # Try to parse JSON if present
                if text_content.startswith("{"):
                    try:
                        parsed_data = json.loads(text_content)
                        if "response" in parsed_data:
                            return {
                                "content": parsed_data["response"],
                                "sources": [service],
                                "confidence": 0.85,
                                "metadata": {
                                    "token_usage": parsed_data.get("token_usage", {}),
                                    "cost_info": parsed_data.get("cost_info", {})
                                }
                            }
                    except json.JSONDecodeError:
                        pass
                
                return {
                    "content": text_content,
                    "sources": [service],
                    "confidence": 0.8,
                    "metadata": {}
                }
            
            # Fallback if no content
            return {
                "content": "اطلاعات دریافت شد اما قابل نمایش نیست.",
                "sources": [service],
                "confidence": 0.3,
                "metadata": {"parse_error": True}
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to parse MCP response: {str(e)}")
            return {
                "content": "خطا در پردازش پاسخ دریافتی.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    def _generate_cache_key(self, service: str, query: str, language: str) -> str:
        """Generate cache key for query"""
        content = f"{service}_{query.lower().strip()}_{language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _find_similar_cached_response(self, service: str, query: str) -> Optional[Dict[str, Any]]:
        """Find similar cached response for fallback"""
        query_words = set(query.lower().split())
        
        for cache_key, cached_data in self.response_cache.items():
            # Simple similarity check based on word overlap
            if service in cache_key:
                # This is a very basic similarity check
                # In production, you might use more sophisticated methods
                return cached_data
        
        return None
    
    def _generate_fallback_response(self, service: str, query: str) -> str:
        """Generate appropriate fallback response"""
        fallback_map = {
            "tabib": "برای دریافت مشاوره پزشکی دقیق، توصیه می‌شود با متخصص مربوطه مشورت کنید. سلامتی شما برای ما مهم است.",
            "f16": "برای راهنمایی در زمینه کسب و کار و کارآفرینی، مطالعه منابع معتبر و مشورت با متخصصان توصیه می‌شود.",
            "digimaman": "در تربیت کودکان، صبر، محبت و درک نیازهای آنها اساسی است. برای راهنمایی تخصصی، با مشاوران کودک مشورت کنید."
        }
        
        return fallback_map.get(service, "متأسفانه در حال حاضر امکان ارائه پاسخ مناسب وجود ندارد.") 