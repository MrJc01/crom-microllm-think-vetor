import ast
import operator as op
import re
import math

class TVDSLInterpreter:
    """
    Interpretador seguro e de alta fidelidade para a Think-Vetor DSL (TV-DSL).
    Intercepta chamadas computacionais dentro das tags de pensamento e as executa deterministicamente.
    """
    
    # Operadores matemáticos permitidos para evitar execução de código Python arbitrário
    SAFE_OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos
    }

    def __init__(self):
        self.variables = {}
        self.history = []
        # Mapeamento de funções funcionais suportadas de forma explícita
        self.functions = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "subtract": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "multiply": lambda a, b: a * b,
            "div": lambda a, b: a / b if b != 0 else "Error: Division by zero",
            "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero",
            "pow": lambda a, b: a ** b,
            "power": lambda a, b: a ** b,
            "sqrt": lambda a: math.sqrt(a) if a >= 0 else "Error: Square root of negative number",
            "abs": lambda a: abs(a),
            "set": lambda name, val: self.set_variable(name, val),
            "get": lambda name: self.get_variable(name),
            "clear_vars": lambda: self.clear_variables(),
            "recall": lambda query: self.recall_context(query)
        }

    def set_variable(self, name, value):
        self.variables[str(name)] = value
        return f"Stored {name} = {value}"
        
    def get_variable(self, name):
        name_str = str(name)
        if name_str in self.variables:
            return self.variables[name_str]
        return f"Error: Variable '{name_str}' is not defined"
        
    def clear_variables(self):
        self.variables.clear()
        return "Variables cleared"
        
    def recall_context(self, query):
        query_lower = str(query).lower()
        for turn in reversed(self.history):
            content = turn.get("content", "")
            if query_lower in content.lower():
                return content
        return "No matching context found"

    def safe_eval(self, expr_str: str):
        """
        Avalia de forma totalmente isolada e segura expressões aritméticas.
        Evita brechas de segurança causadas por eval() clássico.
        """
        # Limpar e normalizar a expressão
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('^', '**') # Converter potenciação clássica para padrão Python
        
        try:
            tree = ast.parse(expr_str, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            return f"Error: Expression parse failure ({str(e)})"

    def _eval_node(self, node):
        # Constantes numéricas
        if isinstance(node, ast.Num): # Python < 3.8 fallback
            return node.n
        elif isinstance(node, ast.Constant): # Python >= 3.8
            return node.value
            
        # Resolução de variáveis (ex: 'x' em 'x + 5')
        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name in self.variables:
                return self.variables[var_name]
            return f"Error: Variable '{var_name}' is not defined"
            
        # Operações binárias (+, -, *, /, **)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(left, str) and left.startswith("Error"):
                return left
            if isinstance(right, str) and right.startswith("Error"):
                return right
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                try:
                    return self.SAFE_OPERATORS[op_type](left, right)
                except ZeroDivisionError:
                    return "Error: Division by zero"
                except Exception as e:
                    return f"Error: operation failed ({str(e)})"
            return f"Error: Unsupported binary operator '{op_type.__name__}'"
            
        # Operações unárias (ex: sinal de menos unário '-5')
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(operand, str) and operand.startswith("Error"):
                return operand
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                return self.SAFE_OPERATORS[op_type](operand)
            return f"Error: Unsupported unary operator '{op_type.__name__}'"
            
        # Chamadas de funções explícitas (ex: multiply(432, 78) ou set(x, 10))
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.functions:
                # Interceptação especial para set(x, valor) para não tentar avaliar 'x' como variável inexistente
                if func_name == "set" and len(node.args) == 2:
                    name_node = node.args[0]
                    var_name = name_node.id if isinstance(name_node, ast.Name) else self._eval_node(name_node)
                    val = self._eval_node(node.args[1])
                    if isinstance(var_name, str) and var_name.startswith("Error"):
                        return var_name
                    if isinstance(val, str) and val.startswith("Error"):
                        return val
                    return self.set_variable(var_name, val)
                
                # Interceptação especial para get(x)
                elif func_name == "get" and len(node.args) == 1:
                    name_node = node.args[0]
                    var_name = name_node.id if isinstance(name_node, ast.Name) else self._eval_node(name_node)
                    if isinstance(var_name, str) and var_name.startswith("Error"):
                        return var_name
                    return self.get_variable(var_name)
                
                # Avaliação padrão de argumentos
                args = [self._eval_node(arg) for arg in node.args]
                for arg in args:
                    if isinstance(arg, str) and arg.startswith("Error"):
                        return arg
                try:
                    return self.functions[func_name](*args)
                except TypeError:
                    return f"Error: Incorrect argument count for '{func_name}'"
                except Exception as e:
                    return f"Error: function execution failed ({str(e)})"
            return f"Error: Function '{func_name}' is not registered in TV-DSL"
            
        # Bloqueio padrão para outros nós AST por motivos de segurança
        return f"Error: AST node type '{type(node).__name__}' is blocked for security"

    def process_text_stream(self, text: str) -> tuple[str, bool]:
        """
        Examina um bloco de texto e processa qualquer tag de comando TV-DSL no formato:
           [TV-DSL: <expressão>]
        Retorna o texto modificado (com os resultados injetados) e uma flag indicando se comandos foram processados.
        """
        # Padrão Regex para encontrar: [TV-DSL: expressão]
        pattern = r"\[TV-DSL:\s*(.*?)\]"
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return text, False
            
        processed_text = text
        offset = 0 # Trata a variação de comprimentos das strings no loop de substituição
        
        for match in matches:
            expr = match.group(1)
            start, end = match.start() + offset, match.end() + offset
            
            # Avaliar a expressão com segurança
            val = self.safe_eval(expr)
            
            # Formatar o resultado de maneira clara e injetável para a LLM ler no contexto
            result_str = f"[TV-DSL: {expr}] -> [RESULT: {val}]"
            
            # Substituir no texto original
            processed_text = processed_text[:start] + result_str + processed_text[end:]
            offset += len(result_str) - (end - start)
            
        return processed_text, True

# Pequeno teste sanitário se executado diretamente
if __name__ == "__main__":
    interpreter = TVDSLInterpreter()
    print("=== TESTE ARITMÉTICA INFIXADA ===")
    print("432 * 78 =", interpreter.safe_eval("432 * 78"))
    print("(150 + 250) / 4 =", interpreter.safe_eval("(150 + 250) / 4"))
    print("2^10 =", interpreter.safe_eval("2^10"))
    
    print("\n=== TESTE ARITMÉTICA FUNCIONAL (DSL) ===")
    print("multiply(432, 78) =", interpreter.safe_eval("multiply(432, 78)"))
    print("sqrt(144) =", interpreter.safe_eval("sqrt(144)"))
    print("divide(10, 0) =", interpreter.safe_eval("divide(10, 0)"))
    
    print("\n=== TESTE DE SEGURANÇA ===")
    print("__import__('os').system('ls') =", interpreter.safe_eval("__import__('os').system('ls')"))
    
    print("\n=== TESTE DE PROCESSAMENTO DE STRING ===")
    raw_thought = "Vamos calcular isso. [TV-DSL: multiply(12, 12)] e depois somar 5. [TV-DSL: 144 + 5]"
    processed, ok = interpreter.process_text_stream(raw_thought)
    print("Original:", raw_thought)
    print("Processado:", processed)
    print("Modificado?", ok)
