from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            expr_load = expr_gen("Load")
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr_at(p, a),
                                       expr_at(c, a),
                                       ]
                        precond_neg = []
                        effect_add = [expr_in(c, p)]
                        effect_rem = [expr_at(c, a)]
                        load = Action(expr_load(c, p, a),
                                      [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        loads.append(load)

            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            expr_unload = expr_gen("Unload")
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr_at(p, a),
                                       expr_in(c, p)]
                        precond_neg = []
                        effect_add = [expr_in(c, a)]
                        effect_rem = [expr_in(c, p)]
                        unload = Action(expr_unload(c, p, a),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        unloads.append(unload)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            expr_fly = expr_gen("Fly")
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr_at(p, fr)]
                            precond_neg = []
                            effect_add = [expr_at(p, to)]
                            effect_rem = [expr_at(p, fr)]
                            fly = Action(expr_fly(p, fr, to),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        new_state = FluentState([], [])
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def expr_gen(a):
    def inner(*args):
        args_list_literal = ', '.join(['{}']*len(args))
        expr_template = f"{a}({args_list_literal})"
        return expr(expr_template.format(*args))
    return inner


def expr_bulk(expr):
    def inner(args_list):
        return [expr(x, y) for x, y in args_list]
    return inner


expr_in = expr_gen("In")
expr_at = expr_gen("At")
expr_load = expr_gen("Load")
expr_bulk_in = expr_bulk(expr_in)
expr_bulk_at = expr_bulk(expr_at)


def air_cargo_p2():
    c1, c2, c3 = 'C1', 'C2', 'C3'
    cargos = {c1, c2, c3}
    p1, p2, p3 = 'P1', 'P2', 'P3'
    planes = {p1, p2, p3}
    jfk, sfo, atl = 'JFK', 'SFO', 'ATL'
    airports = {jfk, sfo, atl}

    pos = [expr_at(c1, sfo),
           expr_at(c2, jfk),
           expr_at(c3, atl),
           expr_at(p1, sfo),
           expr_at(p2, jfk),
           expr_at(p3, atl),
           ]
    neg = (expr_bulk_in([[c, p] for c in cargos for p in planes]) +
           expr_bulk_at([[c1, a] for a in airports - {sfo}]) +
           expr_bulk_at([[c2, a] for a in airports - {jfk}]) +
           expr_bulk_at([[c3, a] for a in airports - {atl}]) +
           expr_bulk_at([[p1, a] for a in airports - {sfo}]) +
           expr_bulk_at([[p2, a] for a in airports - {jfk}]) +
           expr_bulk_at([[p3, a] for a in airports - {atl}]))
    init = FluentState(pos, neg)
    goal = [expr_at(c1, jfk),
            expr_at(c2, sfo),
            expr_at(c3, sfo),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3():
    c1, c2, c3, c4 = 'C1', 'C2', 'C3', 'C4'
    cargos = [c1, c2, c3, c4]
    p1, p2 = 'P1', 'P2'
    planes = [p1, p2]
    jfk, sfo, atl, chi = 'JFK', 'SFO', 'ATL', 'ORD'
    airports = {jfk, sfo, atl, chi}
    pos = [expr_at(c1, sfo),
           expr_at(c2, jfk),
           expr_at(c3, atl),
           expr_at(c4, chi),
           expr_at(p1, sfo),
           expr_at(p2, jfk),
           ]
    neg = (expr_bulk_in([[c, p] for c in cargos for p in planes]) +
           expr_bulk_at([[c1, a] for a in airports - {sfo}]) +
           expr_bulk_at([[c2, a] for a in airports - {jfk}]) +
           expr_bulk_at([[c3, a] for a in airports - {atl}]) +
           expr_bulk_at([[c4, a] for a in airports - {chi}]) +
           expr_bulk_at([[p1, a] for a in airports - {sfo}]) +
           expr_bulk_at([[p2, a] for a in airports - {jfk}]))
    init = FluentState(pos, neg)
    goal = [expr_at(c1, jfk),
            expr_at(c2, sfo),
            expr_at(c3, jfk),
            expr_at(c4, sfo),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


# if __name__ == '__main__':
#     print(air_cargo_p2())
