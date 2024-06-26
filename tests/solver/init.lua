
PrintBuildConfiguration()

ug_load_script("ug_util.lua")
ug_load_script("solver_util/setup_rsamg.lua")

--------------------------------------------------------------------------------
-- Checking for parameters (begin)
--------------------------------------------------------------------------------
-- Several definitions which can be changed by command line parameters
-- space dimension and grid file:

gridName = util.GetParam("-grid", "unit_square_01_tri_2x2.ugx")
numPreRefs = util.GetParamNumber("-numPreRefs", 0, "number of refinements before parallel distribution")
numRefs    = util.GetParamNumber("-numRefs",    9, "number of refinements")

if numPreRefs > numRefs then
	print("It must be choosen: numPreRefs <= numRefs");
	exit();
end

-- parallelisation related stuff
--! way the domain / the grid will be distributed to the processes:
distributionType = util.GetParam("-distType", "bisect", "way the domain will be distributed: [grid2d | bisect | metis]")

--! amount of output
--! set to 0 i.e. for time measurements,
--! >= 1 for writing matrix files etc.
verbosity = util.GetParamNumber("-verb", 0, "verbosity. 0 for time measurements")		

activateDbgWriter = 0	  
--! set to 0 i.e. for time measurements,
--! >= 1 for debug output: call 'set_debug(dbgWriter)'
--! for the main solver ('gmg')
activateDbgWriter = util.GetParamNumber("-dbgw", 0, "debug writer: 0 for no output.") 
						    
bUseAMG = util.HasParamOption("-amg", "use amg")


util.CheckAndPrintHelp("Laplace-Problem\nMartin Rupp");

-- Display parameters (or defaults):
print(" General parameters chosen:")
print("    grid       = " .. gridName)
print("    numRefs    = " .. numRefs)
print("    numPreRefs = " .. numPreRefs)

print("    verb (verbosity)         = " .. verbosity)
print("    dbgw (activateDbgWriter) = " .. activateDbgWriter)

print(" Parallelisation related parameters chosen:")
print("    distType   = " .. distributionType)

--------------------------------------------------------------------------------
-- Checking for parameters (end)
--------------------------------------------------------------------------------

-- choose algebra
InitUG(2, AlgebraType("CPU", 1));

dom=util.CreateAndDistributeDomain(gridName, numRefs, numPreRefs)


--------------------------------------------------------------------------------
-- end of partitioning
--------------------------------------------------------------------------------



-- Make sure, that the required subsets are present
requiredSubsets = {"Inner", "Boundary"}
if util.CheckSubsets(dom, requiredSubsets) == false then 
   print("Subsets missing. Aborting")
   exit()
end

-- write grid to file for test purpose
if verbosity >= 1 then
	refinedGridOutName = "refined_grid_p" .. ProcRank() .. ".ugx"
	print("saving domain to " .. refinedGridOutName)
	SaveDomain(dom, refinedGridOutName)
	
	hierarchyOutName = "hierachy_p" .. ProcRank() .. ".ugx"
	print("saving hierachy to " .. hierarchyOutName)
	if SaveGridHierarchy(dom:grid(), hierarchyOutName) == false then
		print("Saving of domain to " .. hierarchyOutName .. " failed. Aborting.")
		    exit()
	end
end

--print("NumProcs is " .. numProcs .. ", numPreRefs = " .. numPreRefs .. ", numRefs = " .. numRefs .. ", grid = " .. gridName)

-- create Approximation Space
print("Create ApproximationSpace")
approxSpace = ApproximationSpace(dom)
approxSpace:add_fct("c", "Lagrange", 1)
approxSpace:init_levels()
approxSpace:init_top_surface()
approxSpace:print_local_dof_statistic(2)
approxSpace:print_layout_statistic()
approxSpace:print_statistic()

-- lets order indices using Cuthill-McKee
-- OrderCuthillMcKee(approxSpace, true);

--------------------------------------------------------------------------------
--  Assembling
--------------------------------------------------------------------------------
print ("Setting up Assembling")
--------------------------------------------------------------------------------
--  Setup FV Convection-Diffusion Element Discretization
--------------------------------------------------------------------------------

elemDisc = ConvectionDiffusion("c", "Inner", "fv1")
-- LUA User data are not thread safe...
--elemDisc:set_diffusion("Diffusion2d")
--elemDisc:set_velocity("Velocity2d")
--elemDisc:set_reaction_rate("ReactionRate2d")
--elemDisc:set_source("Source2d")
elemDisc:set_diffusion(1.0)
elemDisc:set_source(1.0)


dirichletBND = DirichletBoundary()
dirichletBND:add(1.0, "c", "Boundary")

domainDisc = DomainDiscretization(approxSpace)
domainDisc:add(elemDisc)
domainDisc:add(dirichletBND)