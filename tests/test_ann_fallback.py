import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径，这样才能导入agent模块
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.knowledge_manager import KnowledgeManager, HNSWLIB_AVAILABLE, ANNOY_AVAILABLE, ANN_ENGINE

class TestANNAlgorithmFallback(unittest.TestCase):
    
    def setUp(self):
        # 每个测试用例使用不同的临时数据库文件
        self.test_db_path = f"test_db_{np.random.randint(10000)}.db"
    
    def tearDown(self):
        # 测试完成后删除临时数据库文件
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_default_ann_engine_selection(self):
        """测试默认情况下ANN引擎的选择逻辑"""
        km = KnowledgeManager(db_path=self.test_db_path)
        
        # 验证根据库的可用性，正确选择了ANN引擎
        if HNSWLIB_AVAILABLE:
            self.assertEqual(ANN_ENGINE, "hnswlib")
        elif ANNOY_AVAILABLE:
            self.assertEqual(ANN_ENGINE, "annoy")
        else:
            self.assertEqual(ANN_ENGINE, "linear")
    
    @patch('agent.knowledge_manager.HNSWLIB_AVAILABLE', False)
    def test_hnswlib_unavailable_fallback(self):
        """测试当hnswlib不可用时的降级逻辑"""
        # 在模拟环境中测试
        with patch.dict('agent.knowledge_manager.__dict__', {'HNSWLIB_AVAILABLE': False}):
            # 重新导入以应用模拟
            import importlib
            import agent.knowledge_manager
            importlib.reload(agent.knowledge_manager)
            
            km = agent.knowledge_manager.KnowledgeManager(db_path=self.test_db_path)
            
            # 验证降级到annoy或linear
            if agent.knowledge_manager.ANNOY_AVAILABLE:
                self.assertEqual(agent.knowledge_manager.ANN_ENGINE, "annoy")
            else:
                self.assertEqual(agent.knowledge_manager.ANN_ENGINE, "linear")
    
    @patch('agent.knowledge_manager.HNSWLIB_AVAILABLE', False)
    @patch('agent.knowledge_manager.ANNOY_AVAILABLE', False)
    def test_all_advanced_engines_unavailable(self):
        """测试当所有高级引擎（hnswlib和annoy）都不可用时的降级到linear"""
        with patch.dict('agent.knowledge_manager.__dict__', 
                        {'HNSWLIB_AVAILABLE': False, 'ANNOY_AVAILABLE': False}):
            import importlib
            import agent.knowledge_manager
            importlib.reload(agent.knowledge_manager)
            
            km = agent.knowledge_manager.KnowledgeManager(db_path=self.test_db_path)
            self.assertEqual(agent.knowledge_manager.ANN_ENGINE, "linear")
    
    def test_preferred_engine_selection(self):
        """测试指定首选引擎的功能"""
        # 测试可以指定的引擎
        available_engines = []
        if HNSWLIB_AVAILABLE:
            available_engines.append("hnswlib")
        if ANNOY_AVAILABLE:
            available_engines.append("annoy")
        available_engines.append("linear")
        
        for engine in available_engines:
            with patch.dict('agent.knowledge_manager.__dict__', {'ANN_ENGINE': 'unknown'}):
                km = KnowledgeManager(db_path=self.test_db_path, preferred_ann_engine=engine)
                self.assertEqual(agent.knowledge_manager.ANN_ENGINE, engine)
    
    def test_invalid_preferred_engine(self):
        """测试指定无效首选引擎时的行为"""
        original_engine = ANN_ENGINE
        try:
            km = KnowledgeManager(db_path=self.test_db_path, preferred_ann_engine="invalid_engine")
            # 应该保持原来的引擎
            self.assertEqual(agent.knowledge_manager.ANN_ENGINE, original_engine)
        finally:
            # 恢复原始引擎
            agent.knowledge_manager.ANN_ENGINE = original_engine
    
    def test_vector_search_with_different_engines(self):
        """测试不同引擎下的向量搜索功能"""
        # 添加一些测试数据
        km = KnowledgeManager(db_path=self.test_db_path)
        
        # 使用模拟的嵌入函数，避免真实API调用
        km._get_embedding = MagicMock(return_value=np.random.rand(1536).tolist())
        
        # 添加一些知识项
        km.add_knowledge("测试知识1", "这是第一条测试知识内容")
        km.add_knowledge("测试知识2", "这是第二条测试知识内容")
        km.add_knowledge("测试知识3", "这是第三条测试知识内容")
        
        # 生成查询嵌入
        query_embedding = np.random.rand(1536).tolist()
        
        # 测试向量搜索功能
        results = km.search_vector(query_embedding, limit=2)
        
        # 验证结果格式正确
        self.assertIsInstance(results, list)
        if results:
            # 检查每个结果是否包含必要的字段
            for result in results:
                self.assertIn('id', result)
                self.assertIn('title', result)
                self.assertIn('content', result)
                self.assertIn('similarity', result)
    
    def test_hybrid_search_with_different_engines(self):
        """测试不同引擎下的混合搜索功能"""
        km = KnowledgeManager(db_path=self.test_db_path)
        
        # 使用模拟的嵌入函数
        km._get_embedding = MagicMock(return_value=np.random.rand(1536).tolist())
        
        # 添加测试数据
        km.add_knowledge("测试文档", "这是一份包含测试关键词的文档")
        
        # 执行混合搜索
        results = km.hybrid_search("测试关键词", limit=1)
        
        # 验证结果格式正确
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()