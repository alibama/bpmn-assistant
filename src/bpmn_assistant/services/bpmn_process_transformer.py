from typing import Optional


class BpmnProcessTransformer:
    """
    Class to transform the BPMN process structure generated by the LLM into a new structure more suitable for BPMN XML generation.
    """

    def transform(
        self, process: list[dict], parent_next_element_id: Optional[str] = None
    ) -> dict:
        """
        Restructure the original BPMN JSON structure into a new structure more suitable for BPMN XML generation.

        Example output structure::

            {
                "elements": [
                    {
                        "id": "element_id",
                        "type": "element_type",
                        "label": "element_label",
                        "incoming": ["incoming_flow_id"],
                        "outgoing": ["outgoing_flow_id"]
                    }
                ],
                "flows": [
                    {
                        "id": "flow_id",
                        "sourceRef": "source_element_id",
                        "targetRef": "target_element_id",
                        "condition": "flow_condition"
                    }
                ]
            }
        """

        elements = []
        flows = []

        def handle_exclusive_gateway(
            element: dict, next_element_id: Optional[str] = None
        ) -> Optional[str]:
            # If the exclusive gateway has a 'join' gateway, add it to the elements list
            join_gateway_id = None
            if element.get("has_join", False):
                join_gateway_id = f"{element['id']}-join"
                elements.append(
                    {
                        "id": join_gateway_id,
                        "type": "exclusiveGateway",
                        "label": None,
                    }
                )

            for branch in element["branches"]:
                branch_structure = self.transform(
                    branch["path"], join_gateway_id or next_element_id
                )
                elements.extend(branch_structure["elements"])
                flows.extend(branch_structure["flows"])

                if branch.get("next"):
                    source_ref = element["id"]
                    condition = branch["condition"]
                    if branch_structure["elements"]:
                        source_ref = branch_structure["elements"][-1]["id"]
                        condition = None

                    flows.append(
                        {
                            "id": f"{source_ref}-{branch['next']}",
                            "sourceRef": source_ref,
                            "targetRef": branch["next"],
                            "condition": condition,
                        }
                    )

                # Add the flow from the exclusive gateway to the first element in the branch
                first_element = (
                    branch_structure["elements"][0]
                    if branch_structure["elements"]
                    else None
                )
                if first_element:
                    flows.append(
                        {
                            "id": f"{element['id']}-{first_element['id']}",
                            "sourceRef": element["id"],
                            "targetRef": first_element["id"],
                            "condition": branch["condition"],
                        }
                    )

            return join_gateway_id

        def handle_parallel_gateway(element: dict) -> str:
            # Create a 'join' parallel gateway element
            join_gateway_id = f"{element['id']}-join"
            elements.append(
                {
                    "id": join_gateway_id,
                    "type": "parallelGateway",
                    "label": None,
                }
            )

            for branch in element["branches"]:
                branch_structure = self.transform(branch)
                elements.extend(branch_structure["elements"])
                flows.extend(branch_structure["flows"])

                # Add the flow from the parallel gateway to the first element in the branch
                first_element = branch_structure["elements"][0]
                flows.append(
                    {
                        "id": f"{element['id']}-{first_element['id']}",
                        "sourceRef": element["id"],
                        "targetRef": first_element["id"],
                        "condition": None,
                    }
                )

                # Add the flow from the last element in the branch to the join gateway
                last_element = branch_structure["elements"][-1]
                flows.append(
                    {
                        "id": f"{last_element['id']}-{join_gateway_id}",
                        "sourceRef": last_element["id"],
                        "targetRef": join_gateway_id,
                        "condition": None,
                    }
                )

            return join_gateway_id

        for index, element in enumerate(process):
            next_element_id = (
                process[index + 1]["id"]
                if index < len(process) - 1
                else parent_next_element_id
            )

            elements.append(
                {
                    "id": element["id"],
                    "type": element["type"],
                    "label": element.get("label", None),
                }
            )

            if element["type"] == "exclusiveGateway":
                join_gateway_id = handle_exclusive_gateway(element, next_element_id)

                # Connect the join gateway to the next element in the process
                if join_gateway_id and next_element_id:
                    flows.append(
                        {
                            "id": f"{join_gateway_id}-{next_element_id}",
                            "sourceRef": join_gateway_id,
                            "targetRef": next_element_id,
                            "condition": None,
                        }
                    )
            elif element["type"] == "parallelGateway":
                join_gateway_id = handle_parallel_gateway(element)

                # Connect the join gateway to the next element in the process
                if next_element_id:
                    flows.append(
                        {
                            "id": f"{join_gateway_id}-{next_element_id}",
                            "sourceRef": join_gateway_id,
                            "targetRef": next_element_id,
                            "condition": None,
                        }
                    )
            elif next_element_id:
                # Add the flow between the current element and the next element in the process
                flows.append(
                    {
                        "id": f"{element['id']}-{next_element_id}",
                        "sourceRef": element["id"],
                        "targetRef": next_element_id,
                        "condition": None,
                    }
                )

        # Add incoming and outgoing flows to each element
        for element in elements:
            element["incoming"] = [
                flow["id"] for flow in flows if flow["targetRef"] == element["id"]
            ]
            element["outgoing"] = [
                flow["id"] for flow in flows if flow["sourceRef"] == element["id"]
            ]

        return {"elements": elements, "flows": flows}
