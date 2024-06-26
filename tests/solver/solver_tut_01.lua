--------------------------------------------------------------------------------
--[[!
-- \file apps/conv_diff/laplace.lua
-- \ingroup app_convdiff
-- \{
-- \author Andreas Vogel 
-- \brief Lua - Script to perform the Laplace-Problem
--
]]--
--------------------------------------------------------------------------------
local clock = Chronometer()
clock:tic()
--------------------------------
-- User Data Functions
--------------------------------
function Diffusion2d(x, y, t)
	return	1, 0, 
			0, 1
end

function Velocity2d(x, y, t)
	return	0, 0
end

function ReactionRate2d(x, y, t)
	return	0
end

function Source2d(x, y, t)
	local s = 2*math.pi
	return s*s*(math.sin(s*x) + math.sin(s*y))
end

function DirichletValue2d(x, y, t)
	local s = 2*math.pi
	return true, math.sin(s*x) + math.sin(s*y)
end

ug_load_script("init.lua")

-- creates domainDisc, approxSpace

--------------------------------------------------------------------------------
--  Algebra
--------------------------------------------------------------------------------
print ("Setting up Algebra Solver")


-- create operator from discretization
linOp = AssembledLinearOperator(domainDisc)

-- get grid function
u = GridFunction(approxSpace)
b = GridFunction(approxSpace)


-- create algebraic Preconditioner
jac = Jacobi()
jac:set_damp(0.66)

-- create Convergence Check
convCheck = ConvCheck()
convCheck:set_maximum_steps(1000)
convCheck:set_minimum_defect(1e-11)
convCheck:set_reduction(1e-12)
convCheck:set_verbose(false)

-- create linear solver
linSolver = LinearSolver()
linSolver:set_preconditioner(jac)
linSolver:set_convergence_check(convCheck)

-- create CG solver
cgSolver = CG()
cgSolver:set_preconditioner(jac)
cgSolver:set_convergence_check(convCheck)


-- 0. Reset start solution
u:set(0.0)
print("** Setup (serial):\t".. clock:toc())

os.execute("sleep " .. tonumber(1))
-- 1. init operator
clock:tic()
AssembleLinearOperatorRhsAndSolution(linOp, u, b) 
print("** Assemble matrix:\t".. clock:toc())


-- WARNING_: This measures LUA overhead only!
clock:tic()
for i=1,0 do
	VecNorm(u)
end
print("** VecNorm:\t\t".. clock:toc())

-- WARNING_: This measures LUA overhead only!
clock:tic()
for i=1,0 do
	VecScaleAdd2(u, 0.0, b, -1.0, u)
end
print("** VecScaleAdd:\t\t".. clock:toc())


-- 2. Linear solver (Jacobi)
os.execute("sleep " .. tonumber(1))
clock:tic()
linSolver:init(linOp)
print("** LS init):\t\t"..  clock:toc())
clock:tic()
linSolver:apply_return_defect(u,b)
print("** LS solve:\t".. clock:toc())

-- 2. CG solver (Jacobi)
os.execute("sleep " .. tonumber(1))
clock:tic()
cgSolver:init(linOp)
print("** CG init:\t\t"..  clock:toc())
clock:tic()
cgSolver:apply_return_defect(u,b)
print("** CG solve:\t\t"..  clock:toc())




-- 2. MG solver (Jacobi)


local mg = GeometricMultiGrid(approxSpace)  
mg:set_discretization(domainDisc)

-- Configuration of the smoother.
mg:set_smoother(jac)                  
mg:set_num_presmooth(2)   
mg:set_num_postsmooth(2) 
mg:set_rap(true)          -- RAP is slower

-- Konfiguration Grobgitterloeser
mg:set_base_level(0)        
mg:set_base_solver(LU())     
mg:set_cycle_type("V") 

linSolver:set_preconditioner(mg)
convCheck:set_maximum_steps(10)

os.execute("sleep " .. tonumber(1))
clock:tic()
linSolver:init(linOp)
print("** MG init:\t\t"..  clock:toc())

clock:tic()
linSolver:apply_return_defect(u,b)
print("**  MG apply (10 iter):\t\t"..  clock:toc())

